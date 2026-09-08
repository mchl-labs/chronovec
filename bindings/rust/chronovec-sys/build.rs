// Build the native core from the source bundled in this crate. For local
// development, CHRONOVEC_LIB_DIR remains an escape hatch for linking against
// a separately built shared library.
fn main() {
    println!("cargo:rerun-if-env-changed=CHRONOVEC_LIB_DIR");
    if let Ok(dir) = std::env::var("CHRONOVEC_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=chronovec");
        // The Node addon is distributed with the ChronoVec shared library in
        // the same platform package. Keep the absolute path for local builds,
        // and add a package-relative path for installed addons.
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        } else if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
    } else {
        let destination = cmake::Config::new("native")
            .define("BUILD_SHARED_LIBS", "OFF")
            .build();
        println!(
            "cargo:rustc-link-search=native={}/lib",
            destination.display()
        );
        println!("cargo:rustc-link-lib=static=chronovec");
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
