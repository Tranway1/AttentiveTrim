import datetime
import json

with open("/Users/chunwei/Downloads/llama3-10-15-5k.json", "r") as f:
    data = json.load(f)

res_json = {}
res_json["records"] = []
cnt = 0

for record in data["records"]:
    if cnt == 0:
        cnt += 1
        continue
    start_time = datetime.datetime.strptime(record["start_ts"], "%Y-%m-%d %H:%M:%S.%f")
    prefilling_end_time = datetime.datetime.strptime(record["iterations"][0]["layers"][-1]["iter_ts"], "%Y-%m-%d %H:%M:%S.%f")
    end_time = datetime.datetime.strptime(record["output_ts"], "%Y-%m-%d %H:%M:%S.%f")
    prefilling_runtime = (prefilling_end_time - start_time).total_seconds()
    decoding_runtime = (end_time - prefilling_end_time).total_seconds()

    entry = {"record_id": record["record_id"], "prefilling_runtime": prefilling_runtime,
             "decoding_runtime": decoding_runtime}

    res_json["records"].append(entry)
    cnt += 1

with open("/Users/chunwei/Downloads/llama3-10-15-5k-runtime.json", "w") as f:
    json.dump(res_json, f, indent=4)
