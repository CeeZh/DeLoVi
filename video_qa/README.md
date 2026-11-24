# Question answering on your own video

Answer a question about any video file with DeLoVi-S, the recursive hierarchical search, or DeLoVi-E.

```bash
python -m video_qa.answer video.mp4 "What does the person squeeze onto the plates?"                 # DeLoVi-E
python -m video_qa.answer video.mp4 "What is the cook carrying the platter with?" \
    --options "a folded towel" "oven mitts" "bare hands" "tongs" --method hierarchical_search       # multiple choice
python -m video_qa.answer video.mp4 "What did I just pick up?" --time 00:41:30 --method delovi_s   # asked at 41:30
```

```
Question: What does the person squeeze onto the plates?
Answer (delovi_e): Cauliflower purée  [confidence 0.99]
Evidence: 00:00:00-00:03:00
Input tokens: 2,653 Jev + 1,816 LLM; cost $0.0003 (+ $0.0003 nominal to build the memory tree, cached)
```

The first question on a video builds its memory, the same way as for the benchmarks (`memory.py`):
1. the video is cut into 10-second clips, and GPT-6 Luna captions each clip from one frame per second;
2. the audio is transcribed with the OpenAI Whisper API, and the speech that overlaps a clip is appended to its
   caption after `Speech:`.

The memory is cached in `cache/videos/`, so later questions on the same video only pay for the search and the
answer. Building it costs about $0.4 per hour of video. Requires `ffmpeg` and `ffprobe`, and the API keys of the
main README.

| option | meaning |
|---|---|
| `--method` | `delovi_e` (default), `hierarchical_search` or `delovi_s` |
| `--options A B C ...` | multiple choice; the options are labeled A, B, C, ... (default: open-ended) |
| `--time HH:MM:SS` | the question is asked at this time, so only the video before it is used |
| `--clip-seconds` | length of a clip (default 10) |
| `--translate` | transcribe the speech into English |
| `--description` | what the video is, shown to the models (default: "a video about N minutes long ...") |
| `--dry-run` | fake captions and fake models, to check the setup without API calls |

To build a memory without asking a question: `python -m video_qa.memory video.mp4`.
