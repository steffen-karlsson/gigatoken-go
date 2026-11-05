//! Minimal C-API shim over gigatoken_rs, for embedding in a Go binary.
//!
//! Design notes (see the Go side's README for the full contract):
//!
//! * The handle is an opaque `GigaTokenizer*`; all functions are
//!   `catch_unwind`-guarded so a Rust panic is reported via
//!   [`ggt_last_error`] instead of unwinding across the C boundary (which
//!   would abort the whole Go process).
//! * Single-doc encode goes through the tokenizer's own pretoken cache
//!   (same path as the Python bindings' `encode`) behind a `Mutex`.
//!   Batch encode goes through gigatoken's `WorkerPool` + rayon with an
//!   immutable prototype clone — the pool's documented contract.
//! * `Tokenizer::decode` takes `&[TokenId]`, but `TokenId` lives in a
//!   `pub(crate)` module and is therefore not nameable from outside the
//!   crate. We decode against a copy of the public `vocab_entries()`
//!   table instead, which is byte-identical to upstream's decode (it
//!   just concatenates vocab bytes). If a fork of gigatoken ever makes
//!   `token` public, switch `ggt_decode` to the real `decode` and drop
//!   the table.
//! * Errors are surfaced through a thread-local message,
//!   [`ggt_last_error`], valid until the next shim call on the same
//!   thread.

// FFI entry points: the entire body of each `unsafe extern "C" fn` is
// intentionally unsafe; edition 2024's per-op lint is noise here.
#![allow(unsafe_op_in_unsafe_fn)]

use std::any::Any;
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::ops::Deref;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr::{null, null_mut};
use std::sync::Mutex;

use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use gigatoken_rs::{Tokenizer, WorkerPool, encode_docs_ragged};

/// Batch encode result. `offsets` is cumulative (n_docs + 1 entries), so
/// doc `i`'s ids are `ids[offsets[i]..offsets[i+1]]`. Owned by the caller
/// until passed to [`ggt_free_batch`].
#[repr(C)]
pub struct GgtBatch {
    pub ids: *mut u32,
    pub n_ids: usize,
    pub offsets: *mut i64,
    pub n_offsets: usize,
}

/// Decode result. Owned by the caller until [`ggt_free_bytes`].
#[repr(C)]
pub struct GgtBytes {
    pub data: *mut u8,
    pub len: usize,
}

/// Opaque tokenizer handle.
pub struct GigaTokenizer {
    /// Used for single-doc encode; the `&mut` there only touches the
    /// pretoken cache, never model state.
    single: Mutex<Tokenizer>,
    /// Immutable prototype handed to the WorkerPool for parallel batch
    /// encode. `fork()` shares the vocab/merge tables through `Arc` —
    /// the same call the pool itself uses to fork per-thread workers.
    proto: Tokenizer,
    pool: WorkerPool,
    /// id -> bytes, for decode (see module docs).
    vocab: Vec<Vec<u8>>,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<String>) {
    let msg = msg.into();
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(msg).ok());
}

fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

fn last_error() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().deref() {
        Some(s) => s.as_ptr(),
        None => null(),
    })
}

fn panic_msg(p: Box<dyn Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with non-string payload".to_string()
    }
}

/// Run an FFI body, converting panics into thread-local errors.
fn ffi<R>(f: impl FnOnce() -> Result<R, String>) -> Option<R> {
    clear_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Some(v),
        Ok(Err(msg)) => {
            set_error(msg);
            None
        }
        Err(p) => {
            set_error(panic_msg(p));
            None
        }
    }
}

/// Build the id -> bytes table used by [`ggt_decode`].
fn vocab_table(tok: &Tokenizer) -> Vec<Vec<u8>> {
    let mut table: Vec<Vec<u8>> = Vec::new();
    for (id, bytes) in tok.vocab_entries() {
        let id = id as usize;
        if id >= table.len() {
            table.resize(id + 1, Vec::new());
        }
        table[id] = bytes.to_vec();
    }
    table
}

unsafe fn c_str<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() {
        return Err(format!("{what} is null"));
    }
    CStr::from_ptr(p)
        .to_str()
        .map_err(|e| format!("{what} is not valid UTF-8: {e}"))
}

/// Load a HuggingFace ByteLevel-BPE `tokenizer.json` (GPT-2, RoBERTa, ...)
/// from `path`. Returns NULL on failure; see [`ggt_last_error`].
///
/// # Safety
///
/// `path` must be a valid NUL-terminated C string. The returned handle
/// must be freed exactly once with [`ggt_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_load_hf(path: *const c_char) -> *mut GigaTokenizer {
    ffi(|| {
        let path = c_str(path, "path")?;
        let tokenizer = load_hf_bpe(PathBuf::from(path)).map_err(|e| e.to_string())?;
        let proto = tokenizer.fork();
        let vocab = vocab_table(&tokenizer);
        Ok(Box::into_raw(Box::new(GigaTokenizer {
            single: Mutex::new(tokenizer),
            proto,
            pool: WorkerPool::new(),
            vocab,
        })))
    })
    .unwrap_or(null_mut())
}

/// Free a tokenizer created by [`ggt_load_hf`]. NULL is a no-op.
///
/// # Safety
///
/// `t` must be a handle from [`ggt_load_hf`] that has not been freed
/// before, and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_free(t: *mut GigaTokenizer) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// The tokenizer's vocab size (highest id + 1).
///
/// # Safety
///
/// `t` must be a live handle from [`ggt_load_hf`] (NULL reports an
/// error and returns 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_vocab_size(t: *const GigaTokenizer) -> usize {
    ffi(|| {
        let t = t.as_ref().ok_or("tokenizer handle is null")?;
        Ok(t.proto.vocab_size())
    })
    .unwrap_or(0)
}

/// Encode one document. Returns the number of ids the encode produced
/// (= the number written to `out` when it fits). If `cap` is too small,
/// nothing is written and the needed count is returned, so the caller
/// can retry with a bigger buffer; the retry hits the pretoken cache.
///
/// # Safety
///
/// `t` must be a live handle. `bytes` must be valid for `len` reads
/// (NULL is allowed when `len` is 0); `out` must be valid for `cap`
/// writes, or NULL when `cap` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_encode(
    t: *const GigaTokenizer,
    bytes: *const u8,
    len: usize,
    out: *mut u32,
    cap: usize,
) -> usize {
    ffi(|| {
        let t = t.as_ref().ok_or("tokenizer handle is null")?;
        let input: &[u8] = if bytes.is_null() || len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(bytes, len)
        };
        let mut ids = Vec::with_capacity(len / 4 + 16);
        // Same path as the Python bindings' encode: the mutex is only
        // needed for the &mut the cache requires; output is unaffected.
        let mut tok = t.single.lock().map_err(|_| "tokenizer lock poisoned")?;
        tok.encode_with_added_tokens_flat(input, &mut ids);
        drop(tok);

        let needed = ids.len();
        if needed > 0 && needed <= cap && !out.is_null() {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), out, needed);
        }
        Ok(needed)
    })
    .unwrap_or(usize::MAX)
}

/// Encode `n` documents in parallel with rayon. The documents must be
/// laid out contiguously in `flat` (flat_len bytes), described by `offs`
/// (n + 1 entries): document i is `flat[offs[i]..offs[i+1]]`, with
/// `offs[0] == 0`, `offs[n] == flat_len`, and offsets monotonically
/// non-decreasing. Returns NULL on failure; see [`ggt_last_error`].
/// The result (flat ids + cumulative offsets, n_docs + 1 entries) is
/// owned by the caller; free with [`ggt_free_batch`].
///
/// # Safety
///
/// `t` must be a live handle. `flat` must be valid for `flat_len`
/// reads (NULL when `flat_len` is 0) and `offs` must be valid for
/// `n + 1` reads, satisfying the contract above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_encode_batch(
    t: *const GigaTokenizer,
    flat: *const u8,
    flat_len: usize,
    offs: *const usize,
    n: usize,
) -> *mut GgtBatch {
    ffi(|| {
        let t = t.as_ref().ok_or("tokenizer handle is null")?;
        let flat: &[u8] = if flat.is_null() || flat_len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(flat, flat_len)
        };
        if offs.is_null() {
            return Err("offs is null".to_string());
        }
        let offs = std::slice::from_raw_parts(offs, n + 1);
        if offs[0] != 0 || offs[n] != flat_len {
            return Err(format!(
                "offs[0] must be 0 and offs[n] must equal flat_len ({flat_len})"
            ));
        }
        let mut docs: Vec<&[u8]> = Vec::with_capacity(n);
        for i in 0..n {
            let (start, end) = (offs[i], offs[i + 1]);
            if end < start {
                return Err(format!("offsets not monotonic at doc {i}"));
            }
            docs.push(&flat[start..end]);
        }

        // (flat ids, per-doc lengths) — convert the lengths to offsets.
        let (ids, doc_lens) = encode_docs_ragged(&t.pool, &t.proto, &docs);
        let mut offsets = Vec::with_capacity(doc_lens.len() + 1);
        offsets.push(0i64);
        let mut acc = 0i64;
        for l in &doc_lens {
            acc += *l;
            offsets.push(acc);
        }

        let n_ids = ids.len();
        let n_offsets = offsets.len();
        let ids = ids.into_boxed_slice();
        let offsets = offsets.into_boxed_slice();
        let ids_ptr = Box::into_raw(ids);
        let offsets_ptr = Box::into_raw(offsets);
        Ok(Box::into_raw(Box::new(GgtBatch {
            ids: ids_ptr as *mut u32,
            n_ids,
            offsets: offsets_ptr as *mut i64,
            n_offsets,
        })))
    })
    .unwrap_or(null_mut())
}

/// Free a [`GgtBatch`]. NULL is a no-op.
///
/// # Safety
///
/// `b` must come from [`ggt_encode_batch`], be freed exactly once,
/// and not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_free_batch(b: *mut GgtBatch) {
    if !b.is_null() {
        let b = Box::from_raw(b);
        if !b.ids.is_null() {
            drop(Vec::from_raw_parts(b.ids, b.n_ids, b.n_ids));
        }
        if !b.offsets.is_null() {
            drop(Vec::from_raw_parts(b.offsets, b.n_offsets, b.n_offsets));
        }
    }
}

/// Decode ids back to raw bytes. Returns NULL on failure (e.g. an id out
/// of vocab range); see [`ggt_last_error`]. Free with [`ggt_free_bytes`].
///
/// # Safety
///
/// `t` must be a live handle. `ids` must be valid for `n` reads
/// (NULL when `n` is 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_decode(
    t: *const GigaTokenizer,
    ids: *const u32,
    n: usize,
) -> *mut GgtBytes {
    ffi(|| {
        let t = t.as_ref().ok_or("tokenizer handle is null")?;
        let input: &[u32] = if ids.is_null() || n == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(ids, n)
        };
        let mut out = Vec::new();
        for &id in input {
            let bytes = t.vocab.get(id as usize).ok_or_else(|| {
                format!("token id {id} out of range (vocab size {})", t.vocab.len())
            })?;
            out.extend_from_slice(bytes);
        }
        let len = out.len();
        let ptr = Box::into_raw(out.into_boxed_slice());
        Ok(Box::into_raw(Box::new(GgtBytes {
            data: ptr as *mut u8,
            len,
        })))
    })
    .unwrap_or(null_mut())
}

/// Free a [`GgtBytes`]. NULL is a no-op.
///
/// # Safety
///
/// `b` must come from [`ggt_decode`], be freed exactly once, and not
/// be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ggt_free_bytes(b: *mut GgtBytes) {
    if !b.is_null() {
        let b = Box::from_raw(b);
        if !b.data.is_null() {
            drop(Vec::from_raw_parts(b.data, b.len, b.len));
        }
    }
}

/// The message for the most recent failed call on this thread, or NULL.
/// Valid until the next shim call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn ggt_last_error() -> *const c_char {
    last_error()
}
