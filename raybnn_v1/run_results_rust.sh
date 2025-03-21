#!/bin/bash
#############################
#Generate results for Fig. 1b
#related scripts at:
#RayBNN/examples/figure1b.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure1b --release

#############################
#Generate results for Fig. 2a
#related scripts at:
#RayBNN/examples/figure2a.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure2a --release

#############################
#Generate results for Fig. 2b
#related scripts at:
#RayBNN/examples/figure2b.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure2b --release

#############################
#Generate results for Fig. 2c
#related scripts at:
#RayBNN/examples/figure2c.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure2c --release

#############################
#Generate results for Fig. 2d
#related scripts at:
#RayBNN/examples/figure2d.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure2d --release

#############################
#Generate results for Fig. 2e
#related scripts at:
#RayBNN/examples/figure2e.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure2e --release

#############################
#Generate results for Fig. 2f
#related scripts at:
#RayBNN/examples/figure2f.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure2f --release


#############################
#Generate results for Fig. 4 using RayBNN model
#related scripts at:
#RayBNN model: RayBNN/examples/figure4.rs
#############################
RUSTFLAGS=-Awarnings cargo run --example figure4_raybnn --release



