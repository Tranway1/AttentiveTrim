import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Assuming the model and tokenizer are saved in '~/hf/dbrx-instruct/'
local_model_path = '/home/gridsan/cliu/hf/Mistral-7B-Instruct-v0.2'

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

print(f"Loading model from {local_model_path}")
model = AutoModelForCausalLM.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

messages = [
    {"role": "user", "content": '''
    Here is a paper snippet:
A Deep Dive into Common Open Formats for Analytical DBMSs
Chunwei Liu MIT CSAIL chunwei@csail.mit.edu
ABSTRACT
Anna Pavlenko Microsoft annapa@microsoft.com
Matteo Interlandi Microsoft mainterl@microsoft.com
Brandon Haynes Microsoft brhaynes@microsoft.com
This paper evaluates the suitability of Apache Arrow, Parquet, and ORC as formats for subsumption in an analytical DBMS. We sys- tematically identify and explore the high-level features that are important to support efficient querying in modern OLAP DBMSs and evaluate the ability of each format to support these features. We find that each format has trade-offs that make it more or less suitable for use as a format in a DBMS and identify opportunities to more holistically co-design a unified in-memory and on-disk data representation. Our hope is that this study can be used as a guide for system developers designing and using these formats, as well as provide the community with directions to pursue for improving these common open formats.
PVLDB Reference Format:
Chunwei Liu, Anna Pavlenko, Matteo Interlandi, and Brandon Haynes. A Deep Dive into Common Open Formats for Analytical DBMSs . PVLDB, 16(11): 3044 - 3056, 2023.
doi:10.14778/3611479.3611507
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at https://github.com/Tranway1/ColumnarFormatsEval.
1 INTRODUCTION
Over the last decade, a number of new common and open formats have been proposed that purport to improve performance and ease interoperation across OLAP DBMSs and storage layers such as data lakes [49]. Today, storage formats such as Parquet [12] and ORC [11] are the cornerstone reference architectures for cloud- scale data warehousing systems [14]. At the same time, the in- memory format Apache Arrow [8] is widely considered to be the default means of interoperation across different data systems [50], and several systems are even exploring how to leverage it end-to- end [18]. Each of these in-memory and storage formats attempt to minimize disk, memory, and IO costs, and each applies a wide variety of optimizations to maximize analytic read performance.
Though the benefits for common open formats are now well established [29], there has been less exploration of the relative benefits of these formats for direct subsumption in an analytical DBMS as a native format. This is despite the robust discussion in database community about the relative merits of these formats for this purpose (e.g., [3, 43, 58]). One reason for this is that each for- mat makes design choices that optimize for its use as a common
This work is licensed under the Creative Commons BY-NC-ND 4.0 International License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of this license. For any use beyond those covered by this license, obtain permission by emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 16, No. 11 ISSN 2150-8097. doi:10.14778/3611479.3611507
and open format, and these choices often conflict with longstand- ing analytical DBMS techniques. For example, we increasingly see a push to directly leverage Arrow as an end-to-end, in-memory format within a DBMS [37]. At the same time, DBMS in-memory columnar formats typically encode data [4, 20] to minimize space and reduce memory requirements. However, Apache Arrow by de- fault provides no encoding support, leaving it at odds with typical DBMS design. On the other hand, on-disk formats such as Parquet arrange data in a form that is much closer to that found in modern columnar DBMSs (e.g., by employing run-length encoding mixed with dictionary encoding and bit packing). However, Parquet is widely leveraged only as storage format and exposes no dedicated in-memory representation. Instead, developers bring Parquet data into memory and convert it to the Arrow format, which, as stated above, is suboptimal for a columnar DBMS. Finally, a format such as ORC at first glance appears to offer the best of both worlds, since it provides both an efficiently encoded data format and a related in-memory representation. Nevertheless, Arrow and Parquet are considered the standard nowadays because of their popularity in terms of activity in open-source projects and support from big data frameworks and large-scale query providers [30].
Given this environment, the goal of this paper is to evaluate these three formats, explore their trade-offs, and evaluate their perfor- mance as candidates for direct subsumption in an analytical DBMS. Three main challenges exist in subsuming an in-memory format such as Arrow or traditionally on-disk formats such as Parquet or ORC. First, a DBMS needs to be able to efficiently (de)serialize and (de)compress on-disk data to and from an in-memory representa- tion. For this, efficiency directly depends on format’s compression ratio, decompression speed, and transcoding performance. These trade-offs can be subtle and the line between an “in-memory” or
“on-disk” format is often blurry. For example, in some cases a DBMS could improve performance by writing Arrow to disk or directly operating on Parquet in memory, avoiding transcoding costs and taking advantage of the features offered by each format.
Second, prior work has established that it is highly advantageous for a DBMS to “push down” computation as far as possible (e.g., to disk coprocessors or into the compressed domain) and to do so over as many data types as possible [24, 56]. Computation pushdown subsumes a number of related techniques. Column pruning and data skipping respectively enable a DBMS to avoid decompressing columns or rows that do not contain data relevant to a query answer (e.g., when executing a range query by skipping data regions that do not contain data within the range). Techniques such as direct querying enable a DBMS to retrieve query answers without an expensive decode or decompression step [5, 6, 59]. As we show in Table 1, the ability for common in-memory and on-disk formats to support these techniques is uneven; to maximize performance a DBMS should optimize for the resulting trade-offs.
 3044
  Finally, to maximize performance, modern DBMSs leverage mod- ern techniques such as vectorized execution (e.g., SIMD) [27, 31, 33, 38, 40, 55] and query compilation [19]. Here again the ability for a format to support these techniques is uneven. For example, Parquet can vectorize operations over some query types whereas Arrow is inherently unencoded and more amenable to vectorized execution.
In summary, in this paper we present the first detailed, empirical evaluation of three popular and increasingly-adopted formats and evaluate their suitability to be used as a native format in a DBMS. Our hope is that this study can be used as a guide for system devel- opers using these formats, as well as provide the community with directions to pursue for improving these common open formats.
Columnar Table
Metadata
2
Our contributions include:
• Wesuccinctlysummarizethedesignnuancesanddistinc- tions of three widely-adopted open columnar formats: Apache Arrow, Parquet and ORC (Section 3).
• Wesystematicallyidentifyandexplorethehigh-levelfea- tures that are important to support efficient querying in modern OLAP DBMSs (Section 4).
• Foreachformat,weevaluateitsabilitytosupportefficient encoding, compression, and transcoding (Section 5) for both real-world, synthetic datasets, and various data types.
• Webenchmarktheabilityofeachformattosupportselect- project (SP) operations found near the leaves of query plans. We evaluate these in isolation (Section 6) and in combina- tion (Section 7) using TPC-DS query plan fragments and over various data types.
• Weevaluatetheabilityforeachformattotakeadvantage of recent trends such as vectorization, query compilation, and direct querying (Section 8).
• We identify key opportunities to holistically co-design a unified in-memory and on-disk data representation.
BACKGROUND: COMPRESSION AND DATA
ENCODING
Figure 1: Columnar format layout.
became common, columnar storage formats began to predominate [6]. Columnar databases store data of the same type together [52], allowing systems to leverage lightweight compression, also referred to as encoding [5]. Encoding methods such as dictionary encoding, run-length encoding, and bit-packed encoding are typically designed to compress a specific type of data, enabling efficient compres- sion and better record-level access relative to the general-purpose compression approaches described in the previous section.
Some encoding methods also support direct querying and data skipping to improve query performance [5, 33, 39]. Systems such as Redshift [26] and SQL Server [35] support many lightweight compression approaches that reduce the storage cost of data; at the same time, they apply compression-specific optimizations to improve query execution performance. Previous research [16, 32, 40, 41] has also demonstrated that, for specific datasets, good encoding achieves a comparable compression ratio with far fewer CPU cycles than does byte-oriented compression algorithms.
We next give an overview of several popular encoding algorithms referred to in this paper and highlight their applicable scenarios. Bit-Packed Encoding (BP) works on numerical data. It finds the minimal number of bits needed to represent values and removes superfluous leading zeros. It works best when the target numbers have similar bit-width.
Dictionary Encoding (DICT) works on all data types. It encodes each distinct entry with an integer key and bit-packs the integers. Dictionary encoding works best when the dataset has small cardi- nality and many repetitions. Queries on dictionary encoded data can be applied either on the fully decoded data or directly in the encoded domain after query rewriting using dictionary translation. Run-Length Encoding (RLE) works on data with many consec- utive repetitions. It replaces a run of the same value with a pair consisting of the value and how many times it is repeated. Hybrid Encodings are derived from the above encoding tech- niques. Dictionary run-length encoding (DICT-RLE) applies RLE on the dictionary encoded keys to further compress data. Bit-packed and run-length hybrid encoding is used as a default implementa- tion for the Parquet RLE encoder. Hybrid encoding usually achieves better compression performance at the cost of performance.
3 COLUMNAR OPEN FORMATS
In big data environments today, there are many optimized data formats for columnar data storage and computation, as we show in Table 1. Interestingly, these data formats share the same basic
Data systems employ compression algorithms to reduce on-disk or in-memory data sizes and improve bandwidth utilization [25]. Conventional compression has traditionally focused on minimizing file size. This focus on size alone, while appropriate for storage, overlooks DBMS query execution performance [33, 46, 51]. Con- versely, in an analytical columnar DBMS, compressed size is usually balanced with the ability to query directly on the compressed data.
2.1 Compression
Because of their generality, byte-oriented compression techniques (e.g., Gzip [17], Snappy [23], and Zlib [22]) are widely used to re- duce data size [5, 46]. They treat the input values as a byte stream and compress them sequentially. Byte-oriented compression is ap- plicable to all data types and, in general, exhibits good compression ratios [33]. However, these methods are computationally inten- sive [5]. A data block needs to be fully decompressed before indi- vidual values can be accessed. This often introduces unnecessary overhead for query execution.
2.2 Encoding
For decades, many data engines used row-oriented storage formats for OLTP query workloads [3]. As more complex OLAP workloads
3045
     Row Batch 1
Chunked Chunked ... Chunked Column 1 Column 2 Column m
 Row Batch 2
  ...
   Row Batch n

Table 1: A comparison of the features found in common open columnar data formats.
  Arrow Feather Parquet ORC
Encoding Methods
DICT
DICT
DICT(-RLE), RLE, BP, Delta, etc. DICT, RLE, BP, Delta
Compression Codecs
None
Zstd, LZ4
Gzip, Snappy, Zstd, LZ4, (LZO) Snappy, Zlib, LZ4
Skipping
Chunk-level None Record-level Chunk-level
Direct Query
None None None None
Primary Purpose
In-Memory Compute On-Disk Storage On-Disk Storage On-Disk Storage
Representative Systems
Dremio, Spark, Pandas, etc. Pandas
Spark, Hive, Presto, etc. Hive, Presto, etc.
       Dictionary Page
    Index data column n:
Min/Max values. Row position. Block offsets. Bloom filter.
Row data column n with Integer:
Present (non-null) bit stream. Integers encoded as RLE/Bitpacking. /////////////////////
Row data column n with String: Present (non-null) bit stream. Dictionary data (bytes).
Now please answer the questions: What is contribution of the paper?
    Please only answer the question I asked above and print the answer in a single line.'''}

    # ,
    # {"role": "user", "content": "What is your favourite condiment?"},
    # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    # {"role": "user", "content": "Do you have mayonnaise recipes?"}
]


start = time.time()

print("Sending model to device...")
model.to(device)
send_model = time.time() - start
print(f"Duration: {send_model:.2f} seconds")

print("Tokenizing messages...")
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
tokenizing = time.time() - start
print(f"Duration: {tokenizing:.2f} seconds")
print("Sending data to device...")
model_inputs = encodeds.to(device)

sending_data = time.time() - start
print(f"Duration: {sending_data:.2f} seconds")
print("Generating response...")
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
generate_res = time.time() - start
print(f"Duration: {generate_res:.2f} seconds")
print("Decoding response...")
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
decoding_res = time.time() - start
print(f"Duration: {decoding_res:.2f} seconds")
print(f"Answer:{decoded[0]}")

print(f"send_model, tokenizing, sending_data, generate_res, decoding_res")
print(f"{send_model:.2f}, {tokenizing-send_model:.2f}, {sending_data-tokenizing:.2f}, {generate_res-sending_data:.2f}, {decoding_res-generate_res:.2f}")
