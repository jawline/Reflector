import os
import sys
import math
import subprocess
import torch
from preprocess import TerrainDatasetSlow


def worker(samples_dir, output_dir, start, end):
    dataset = TerrainDatasetSlow(samples_dir)
    remaining = end - start
    print(f"total={remaining}")
    for i in range(start, end):
        remaining -= 1
        element = dataset[i]
        if not element["broken"]:
            element["terrain"] = element["terrain"].unsqueeze(0)
            element["mask"] = element["mask"].unsqueeze(0)
            torch.save(element, f"{output_dir}/{i}.pt")
        print(f"i={i} remaining={remaining}")


def main():
    from_ = sys.argv[-3]
    to_ = sys.argv[-2]
    num_workers = int(sys.argv[-1])

    total = len(TerrainDatasetSlow(from_))
    chunk_size = math.ceil(total / num_workers)

    processes = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min(start + chunk_size, total)
        log_path = f"{to_}/worker_{w}.log"
        log_file = open(log_path, "w")
        p = subprocess.Popen(
            [
                sys.executable, __file__,
                "--worker", from_, to_, str(start), str(end),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.close()
        processes.append((p, log_path))

    for p, log_path in processes:
        p.wait()

    print("All workers finished")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        idx = sys.argv.index("--worker")
        worker(sys.argv[idx + 1], sys.argv[idx + 2],
               int(sys.argv[idx + 3]), int(sys.argv[idx + 4]))
    else:
        main()
