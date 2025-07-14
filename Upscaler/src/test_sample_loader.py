from sample_loader import load_sample

loaded = load_sample("./test_sample.h3t")

print("Width:", loaded.width)
print("Height:", loaded.height)
print("Scale Z", loaded.scale_z)

for elt in loaded.data:
    print(elt)
