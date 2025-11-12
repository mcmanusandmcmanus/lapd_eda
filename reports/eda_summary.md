# LAPD Calls for Service — Inspection & EDA

## Dataset Snapshot
- **Records analysed:** 2,760,470 (2,760,470 unique incidents)
- **Temporal coverage:** 01 Jan 2024 to 01 Nov 2025 (~670 days)
- **Dispatch timestamps present:** 100.0%
- **Most active areas:** Outside (646,012), Central (171,912), Newton (134,472), 77th Street (129,608), Southwest (126,226)
- **Top call types:** Code 6 (1,273,792), 902 Traffic Stop (186,476), 906 Code 30 Ringer (79,538), 415 Man (69,455), 921 Trespass Susp (38,287)

## Data Quality Watchlist
- `Rpt_Dist` missing 23.4%

## Operational Highlights
- Daily call demand ranges between 347 and 5,138 with a median of 4,183 calls per day.
- Peak dispatch hour: 19:00 with 149,508 calls; quietest hour is 04:00 (46,705 calls).
- Friday carries 15.1% of all calls, outpacing the slowest day (Sunday at 12.7%).
- "Outside" calls account for 23.4% of all dispatches, indicating notable mutual-aid or cross-jurisdiction events.
- CODE 6 related activity represents 46.1% of calls, underscoring the load of investigative/stakeout work.

## Files Generated
- Clean call-level parquet: `C:\Users\mcman\webapp_lapd_datascience\data\interim\lapd_calls_clean.parquet`
- Feature-enhanced parquet: `C:\Users\mcman\webapp_lapd_datascience\data\processed\lapd_calls_features.parquet`
- Area-hour aggregate: `C:\Users\mcman\webapp_lapd_datascience\data\processed\call_volume_area_hour.parquet`
- Daily call volume: `C:\Users\mcman\webapp_lapd_datascience\data\processed\daily_call_volume.parquet`
- Call type summary: `C:\Users\mcman\webapp_lapd_datascience\data\processed\call_type_summary.parquet`
- Visuals directory: `C:\Users\mcman\webapp_lapd_datascience\reports\figures`