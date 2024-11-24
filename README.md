# case-study-hr
Q&amp;A solution for a fictional HR department.
Completely open source. See more here: https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/

## Installation notes
Installing llama-cpp to work with METAL processors requires adding the environment variables
´´´CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1´´´ before poetry install command 
