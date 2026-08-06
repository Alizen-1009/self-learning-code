# Lesson 9: GDN Capstone

Do not read `grade.py` before your first attempt.

## Part A — Code, 6 points

Implement both TODOs in `challenge.py`, then run:

```bash
python3 grade.py
```

Tasks:

1. one V-first recurrent GDN step, without mutating the input state — 4 points;
2. FLA/FlashInfer scheduling counts — 2 points.

## Part B — Written, 4 points

Answer each in 2–5 sentences:

1. Why does FLA save both `H[chunk]` and `R/v_new`, while FlashInfer SM103 normally does not?
2. For FLA, which stage is chunk-serial, and which dimensions remain parallel there?
3. What is the main state-storage/scheduling difference between ordinary FlashInfer SM90 and SM103 GDN prefill?
4. Given the Lesson-7 materialized pipeline, what evidence would you collect before choosing the first fusion optimization?

## Pass criterion

```text
8 / 10
```

After passing, send the auto score and four written answers to the teacher. GDN will be recorded as a completed milestone and the course will move to KDA.
