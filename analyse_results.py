import json

with open("results.json", "r") as f:
    stats = json.load(f)


for n_receivers in stats.keys():
    for approx in [True, False]:
        key = str(approx).lower()
        got = stats[n_receivers][key]
        tot = sum(1 for b in got if b)
        stats[n_receivers][key] = tot / len(got)
print(stats)

with open("results.json", "r") as f:
    stats = json.load(f)

for n_receivers in stats.keys():
    count = 0
    total = 0
    n = len(stats[n_receivers]["true"])
    for i in range(n):
        if stats[n_receivers]["false"][i]:
            total += 1
            if stats[n_receivers]["true"][i]:
                count += 1
    stats[n_receivers] = count / total
print(stats)
