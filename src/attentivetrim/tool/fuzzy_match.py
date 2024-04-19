import json

from fuzzywuzzy import process, fuzz


def best_substring_match(query, string):
    # This will extract all substrings of length equal to the query from the string
    candidates = [string[i:i + len(query)] for i in range(len(string) - len(query) + 1)]
    print("grd:", query)
    # Find the best match among the candidates
    ret = process.extractOne(query, candidates, scorer=fuzz.ratio)
    if ret is None:
        return None

    best_match, score = ret
    positions = [can == best_match for can in candidates]
    start = positions.index(True)
    end = start + len(query)
    print("best match:", best_match, "score:", score, "start:", start, "end:", end)
    # print("-------", string[start:end])
    return start, end


if __name__ == "__main__":
    grd_file ="../data/test_v16_inputfile100-result-What is the aut-0.1.json"
    res_file = grd_file.replace(".json", "-location.json")
    with open(grd_file) as f:
        json_obj = json.loads(f.read())
    res_obj = {}
    res_obj["question"] = json_obj["question"]
    res_obj["files"] = []
    list_of_file_extraction = json_obj["files"]
    for extraction  in list_of_file_extraction:
        file_path = extraction["file"]
        query = extraction["groundtruth"]
        with open('/Users/chunwei/pvldb_1-16/16/' + file_path) as f_in:
            doc_dict = json.load(f_in)
        string = doc_dict["symbols"]
        total_chars = len(string)
        start, end = best_substring_match(query, string)
        res_obj["files"].append({"file": file_path, "total_chars": total_chars,  "start": start, "end": end})
        print("file:", file_path, " total_chars: ", total_chars, " start: ", start, " end: ", end)
    with open(res_file, "w") as f:
        f.write(json.dumps(res_obj, indent=4))
