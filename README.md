# cs2-de

> [!NOTE]
> This project is still in progress.

<img width="900" height="626" alt="image" src="https://github.com/user-attachments/assets/31070b23-9499-416c-b57f-bed98dcec9de" />



The competitive Counter-Strike 2 (CS2) scene generates gigabytes of data per match in the form of .dem files (tick-by-tick replays). However, raw data doesn't win championships. This project is a Data Engineering and Analytics Pipeline focused on transforming massive Source Engine logs into actionable tactical intelligence for Coaches and In-Game Leaders (IGLs) of eSports teams (Tier 1 to Tier 3).

## Data Architecture

🥉 Raw data extraction from the .dem binary file using Python/Go parsers. The data here contains all the noise from the game engine (warm-up phase, server resets, etc.).

🥈 Rigorous cleaning, application of Data Contracts (strongly typed schemas), and conversion to Parquet format (high columnar compression).

Applied Business Rules: Filtering of server-caused deaths (world entity in Tick 5), removal of Warmup data, and standardization of relationship keys (match_id).

🥇 Under Development: Cross-referencing of fact and dimension tables to generate high-value metrics (e.g., Kills enriched with the context of the Rounds table).



> [!TIP]
> If you have any difficulty understanding the premise or replicating the project, please contact me directly.

