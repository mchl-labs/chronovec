extern crate napi_build;

fn main() {
    println!("cargo:rerun-if-env-changed=CHRONOVEC_LIB_DIR");
    if let Ok(dir) = std::env::var("CHRONOVEC_LIB_DIR") {
        // The publish workflow places the shared ChronoVec runtime beside the
        // addon in each platform package. Keep an absolute path for local
        // builds and add a package-relative path for installed addons.
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{dir}");
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,@loader_path");
        } else if cfg!(target_os = "linux") {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{dir}");
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN");
        }
    }
    napi_build::setup();
}
