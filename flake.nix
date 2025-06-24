{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
    rust-overlay = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:oxalica/rust-overlay";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.rust-overlay.overlays.default ];
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        python-env = pkgs.python3.withPackages
          (ps: with ps; [ sqlalchemy pyarrow datasets ipdb ]);
      in {
        devShell = pkgs.mkShell rec {
          nativeBuildInputs = with pkgs; [
            cudatoolkit
            cudaPackages.cuda_cudart.dev
            cudaPackages.cudnn.dev
            openssl.dev
            pkg-config
            cargo-watch
            # libtorch-bin
            python-env
            rust-bin.stable."1.86.0".default
            rust-analyzer
            stdenv.cc.cc.lib
            sqlite
            wgpu-utils
            vulkan-loader
            vulkan-headers
            glslang
          ];
          CUDA_TOOLKIT_ROOT_DIR = pkgs.cudatoolkit.out;
          CUDA_PATH = pkgs.cudatoolkit.out;
          CUDNN_LIB = pkgs.cudaPackages.cudnn.dev;
          LD_LIBRARY_PATH = "${pkgs.addDriverRunpath.driverLink}/lib:${
              pkgs.lib.makeLibraryPath nativeBuildInputs
            }";
          INCLUDE_PATH = pkgs.lib.makeIncludePath nativeBuildInputs;
        };
      });
}
