from unicore.data import LMDBDataset

path = "./data/DUD-E/raw/all/ada17/mols.lmdb"  # 换成一个具体 target
print("open", path)
ds = LMDBDataset(path)
print("len =", len(ds))

for i in range(len(ds)):
    if i % 1000 == 0:
        print("index", i, "...")
    ds[i]   # 只要能成功 __getitem__ 就行
print("done")