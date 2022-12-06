import pandas as pd
from matplotlib import pyplot as plt

OUTPUT = "./problem_e_output.txt"

with open(OUTPUT, "r") as f:
    lines = f.readlines()

chunks = []
for line in lines:
    if line.startswith("+ ./target/release/smooth-image"):
        chunks.append([])
    if chunks:
        chunks[-1].append(line)

run_info = [line for c in chunks for line in c if line.startswith("+ ./target/release/smooth-image")]
algo = [line.split("--algo")[1][1:3].upper() for line in run_info]
pic_info = [line.split()[4].split("/")[-1].split(".")[0] for line in run_info]
iter_data = [[line.split(", ")[:2] for line in c if line.startswith("iter=")] for c in chunks]
iter_data = [pd.DataFrame([{i.split("=")[0]: float(i.split("=")[1]) for i in line} for line in c]) for c in iter_data]

for d in iter_data:
    d.set_index("iter", inplace=True)

plot_dict = {}
for alg, pic, df in zip(algo, pic_info, iter_data):
    if pic not in plot_dict:
        plot_dict[pic] = {}
    plot_dict[pic][alg] = df


def first_index(lis, thresh=3.5):
    for i, n in enumerate(lis):
        if n > thresh:
            return i
    return len(lis)


threshs = {
    "512_512_pens": 3.45,
    "1024_1024_bluestreet": 3.5,
    "4096_4096_husky": 3.5,
}

plot_to = {
    "512_512_pens": 1000,
    "1024_1024_bluestreet": 600,
    "4096_4096_husky": 1400,
}


for pic, data in plot_dict.items():
    cg: pd.DataFrame = data["CG"]
    ag: pd.DataFrame = data["AG"]
    merged = pd.merge(cg, ag, how="outer", left_index=True, right_index=True)
    merged.columns = ["CG", "AG"]
    cg_first_3_5 = merged.index[first_index(merged["CG"], thresh=threshs[pic]) - 1]
    ag_first_3_5 = merged.index[first_index(merged["AG"], thresh=threshs[pic]) - 1]
    merged = merged.loc[merged.index < plot_to[pic], :]
    merged.plot(figsize=(6, 4))
    plt.title(f"{pic} psnr")
    plt.vlines(x=cg_first_3_5, ymin=0, ymax=3.6, colors="blue",
               label=f"CG psnr reaches {threshs[pic]} at iter {int(cg_first_3_5)}")
    plt.vlines(x=ag_first_3_5, ymin=0, ymax=3.6, colors="orange",
               label=f"AG psnr reaches {threshs[pic]} at iter {int(ag_first_3_5)}")
    plt.grid()
    plt.legend()
    plt.savefig(f"./plots/{pic}-analysis.png", dpi=300)
