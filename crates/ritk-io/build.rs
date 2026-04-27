//! Build script for ritk-io.
//!
//! # Purpose
//!
//! Links the C++ standard library (`libstdc++`) when building for GNU targets,
//! and ensures the linker search path includes the directory containing
//! `libstdc++.a` so that `lld` (used via `-fuse-ld=lld` on Windows GNU) can
//! resolve C++ runtime ABI symbols.
//!
//! # Why this is required
//!
//! `charls-sys` compiles CharLS (ISO 14495-1 / ITU-T T.87 JPEG-LS reference
//! implementation) as a static C++ library (`libcharls.a`). The compiled archive
//! contains unresolved references to the C++ runtime ABI symbols:
//!
//! - `__cxa_allocate_exception` / `__cxa_free_exception` — C++ exception allocation
//! - `__cxa_throw` / `__cxa_begin_catch` / `__cxa_end_catch` — exception dispatch
//! - `__cxa_rethrow` — re-throw
//! - `__gxx_personality_seh0` — SEH-based stack unwinding personality (MinGW)
//! - `_Unwind_Resume` — DWARF/SEH unwind resume
//! - `std::system_error`, `std::runtime_error` — C++ standard exception types
//!
//! Rust's linker invocation passes `-nodefaultlibs`, which disables the automatic
//! inclusion of `libstdc++`. On GNU targets, `libstdc++` must therefore be linked
//! explicitly. Without it, the final binary link fails with undefined symbol errors.
//!
//! # Search path for lld on Windows GNU (MSYS2)
//!
//! When `-fuse-ld=lld` is active (see `.cargo/config.toml`), `lld` does not
//! automatically inherit the GCC library search directories (e.g.
//! `D:/msys64/ucrt64/lib`). This build script detects the location of
//! `libstdc++.a` by querying `g++ -print-file-name=libstdc++.a` (or the compiler
//! named by the `CXX` environment variable) and emits
//! `cargo:rustc-link-search=native=<dir>` so that `lld` can find the archive.
//!
//! The detection is best-effort: if the query fails (compiler absent in CI
//! environments that use pure Clang), the search-path directive is omitted and
//! the system linker search path must cover `libstdc++.a` through other means
//! (e.g. `/usr/lib/x86_64-linux-gnu` on Debian/Ubuntu is already in the default
//! path).
//!
//! # Invariant
//!
//! `cargo:rustc-link-lib=stdc++` is emitted only for `target_env = "gnu"`.
//! `cargo:rustc-link-lib=c++` is emitted only for `target_os = "macos"`.
//! On MSVC the C++ runtime is linked automatically; no directive is needed.

fn main() {
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_env == "gnu" {
        // Detect libstdc++ location and add it to the linker search path.
        // This is required for lld on MSYS2/Windows GNU, where lld does not
        // inherit GCC's library search directories automatically.
        if let Some(dir) = locate_libstdcxx_dir() {
            println!("cargo:rustc-link-search=native={}", dir);
        }

        // charls-sys (C++) and openjpeg-sys (C) both pull in this crate's link
        // directives. Linking libstdc++ for a C-only consumer is harmless;
        // --gc-sections eliminates unreferenced symbols from the final binary.
        println!("cargo:rustc-link-lib=stdc++");
    } else if target_os == "macos" {
        // On macOS with AppleClang / LLVM, the C++ runtime is `libc++`.
        // charls-sys compiled with AppleClang uses the libc++ ABI.
        println!("cargo:rustc-link-lib=c++");
    }
    // MSVC: the MSVC C++ runtime is linked automatically by Rust's linker driver.
}

/// Query the C++ compiler for the full path of `libstdc++.a` and return its
/// parent directory as a `String`, or `None` when the query cannot be completed.
///
/// # Algorithm
///
/// Tries compilers in priority order:
/// 1. The compiler named by the `CXX` environment variable (if set).
/// 2. `g++`
/// 3. `clang++`
///
/// For each candidate, runs `<compiler> -print-file-name=libstdc++.a`.
/// The output is the full absolute path when the file exists, or the bare
/// filename `libstdc++.a` when it does not. The function returns the parent
/// directory only when an absolute path is obtained.
fn locate_libstdcxx_dir() -> Option<String> {
    let candidates: Vec<String> = {
        let mut v = Vec::new();
        if let Ok(cxx) = std::env::var("CXX") {
            v.push(cxx);
        }
        v.push("g++".to_owned());
        v.push("clang++".to_owned());
        v
    };

    for compiler in &candidates {
        if let Some(dir) = query_libstdcxx(compiler) {
            return Some(dir);
        }
    }
    None
}

/// Run `<compiler> -print-file-name=libstdc++.a` and return the parent directory
/// of the reported path, or `None` when the output is not an absolute path.
fn query_libstdcxx(compiler: &str) -> Option<String> {
    let output = std::process::Command::new(compiler)
        .arg("-print-file-name=libstdc++.a")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let raw = std::str::from_utf8(&output.stdout).ok()?.trim().to_owned();

    // If the compiler cannot find the file, it echoes the bare filename.
    // An absolute path starts with '/' on POSIX or a drive letter on Windows.
    let path = std::path::Path::new(&raw);
    if !path.is_absolute() {
        return None;
    }

    // Canonicalize to resolve /../ components that lld cannot normalize itself.
    let parent = path.parent()?;
    let dir_str = match parent.canonicalize() {
        Ok(canonical) => canonical.to_string_lossy().into_owned(),
        Err(_) => parent.to_string_lossy().into_owned(),
    };
    // Windows canonicalize() prepends "\\?\" (extended-length path prefix).
    // lld does not accept this prefix in -L search paths; strip it when present.
    let dir_str = dir_str.strip_prefix(r"\\?\").unwrap_or(&dir_str).to_owned();
    Some(dir_str)
}
