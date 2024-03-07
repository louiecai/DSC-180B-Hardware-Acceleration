```angular2html
env 1-4 are testing the increase in vcpus while scaling memory proportionally.
env 5-7 hold the number of vcpus constant at 2, while scaling the memory to see its direct relationship.
env 8-13 introduce different gpus to measure impact while attempting to hold vcpus and memory constant using available instance configurations.
```

| Environment | Instance     | vCPU | CPU Memory | CPU Type                 | GPU | GPU Memory | GPU Type         | done |
|-------------|--------------|------|------------|--------------------------|-----|------------|------------------|------|
| env1        | c6g.medium   | 1    | 2.0 GiB    | AWS Graviton2            | 0   | N/A        | N/A              ||
| env2        | c6g.large    | 2    | 4.0 GiB    | AWS Graviton2            | 0   | N/A        | N/A              ||
| env3        | c6g.xlarge   | 4    | 8.0 GiB    | AWS Graviton2            | 0   | N/A        | N/A              ||
| env4        | c6g.2xlarge  | 8    | 16.0 GiB   | AWS Graviton2            | 0   | N/A        | N/A              ||
| env5        | c1.medium    | 2    | 1.7 GiB    | Intel Xeon E5-2670       | 0   | N/A        | N/A              ||
| env6        | r5.large     | 2    | 16.0 GiB   | Intel Xeon Platinum 8000 | 0   | N/A        | N/A              ||
| env7        | x2gd.large   | 2    | 32.0 GiB   | AWS Graviton2            | 0   | N/A        | N/A              ||
| env8        | g3s.xlarge	  | 4    | 30.5	GiB   | Intel Xeon E5 2686 v4    | 1   | 8 GiB      | NVIDIA Tesla M60 ||
| env9        | g4dn.xlarge  | 4    | 16 GiB     | Intel Cascade Lake       | 1   | 16 GiB     | NVIDIA T4        ||
| env10       | g5.xlarge    | 4    | 16 GiB     | AMD EPYC Gen2            | 1   | 24 GiB     | NVIDIA A10G      ||
| env11       | g3.4xlarge	  | 16   | 122 GiB    | Intel Xeon E5 2686 v4    | 1   | 8 GiB      | NVIDIA Tesla M60 ||
| env12       | g4dn.4xlarge | 16   | 64 GiB     | Intel Cascade Lake       | 1   | 16 GiB     | NVIDIA T4        ||
| env13       | g5.4xlarge   | 16   | 64 GiB     | AMD EPYC Gen2            | 1   | 24 GiB     | NVIDIA A10G      ||
| datahub1       | N/A   | 1 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218          | 1   | N/A        | 2080ti        |1 trial|
| datahub2      | N/A   | 2 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218            | 1   | N/A        | 2080ti          |1 trial|
| datahub3       | N/A   | 4 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218             | 1   | N/A        | 2080ti        |1 trial|
| datahub4      | N/A   | 8 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218           | 1   | N/A        | 2080ti          |1 trial|
| datahub5       | N/A   | 4 core    | 8.0 GiB    | xxx            | 1   | N/A        | 2080ti        ||
| datahub6      | N/A   | 4 core    | 8.0 GiB    | xxx          | 1   | N/A        | a5000          ||
