//! Build script for ritk-io.
//!
//! # Purpose
//!
//! Links the C++ standard library (`libstdc++`) when building for GNU targets.
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
//!
//! Rust's linker invocation passes `-nodefaultlibs`, which disables the automatic
//! inclusion of `libstdc++`. On GNU targets, `libstdc++` must therefore be linked
//! explicitly. Without it, the final test-binary link fails with:
//!
//! ```text
//! collect2.exe: error: ld returned 5 exit status
//! ```
//!
//! (exit code 5 = Windows ERROR_ACCESS_DENIED emitted by `collect2` when `ld.exe`
//! cannot resolve relocations and aborts before writing the output file).
//!
//! # Invariant
//!
//! This directive is emitted only for `target_env = "gnu"` (MinGW/MSYS2 and
//! Linux GNU toolchains). On MSVC and macOS targets, the C++ runtime is either
//! automatically included or provided via a different mechanism (`c++` on macOS).

fn main() {
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_env == "gnu" {
        // charls-sys (C++) requires libstdc++ for exception handling ABI symbols.
        // openjpeg-sys (C) does not require this; linking libstdc++ for C-only code
        // is harmless (unused symbols are eliminated by --gc-sections).
        println!("cargo:rustc-link-lib=stdc++");
    } else if target_os == "macos" {
        // On macOS with the LLVM/Clang toolchain, the C++ runtime is `libc++`.
        // charls-sys compiled with AppleClang uses the libc++ ABI, not libstdc++.
        println!("cargo:rustc-link-lib=c++");
    }
    // MSVC: the MSVC C++ runtime is linked automatically by Rust's linker driver;
    // no explicit directive is needed.
}
