# NovaBank Data Dictionary (Bank Marketing — Final Run)
Short, plain-language descriptions (1–2 lines per field). Includes a note where modeling treatment differs (e.g., dropped or encoded).

> Source file: `bank-additional-full.csv` (UCI Bank Marketing).  
> Target we used in modeling: `target = 1 if y == "yes" else 0`.

| Field | Type | Description | Notes / Modeling Treatment |
|---|---|---|---|
| age | numeric | Customer age in years. | Used as-is; standardized for modeling. |
| job | categorical | Main job type (e.g., admin., blue-collar, technician, student). | One‑hot encoded. |
| marital | categorical | Marital status (single, married, divorced). | One‑hot encoded. |
| education | categorical | Highest education level (e.g., basic.4y, high.school, university.degree). | One‑hot encoded. |
| default | categorical | Has credit in default? (yes/no/unknown). | One‑hot encoded; treat “unknown” as its own level. |
| housing | categorical | Has a housing loan? (yes/no/unknown). | One‑hot encoded. |
| loan | categorical | Has a personal loan? (yes/no/unknown). | One‑hot encoded. |
| contact | categorical | Contact communication type (cellular or telephone). | One‑hot encoded. |
| month | categorical | Last contact month (jan–dec). Captures seasonal/timing effects. | One‑hot encoded. |
| day_of_week | categorical | Last contact weekday (mon–fri). Captures weekday effects. | One‑hot encoded. |
| duration | numeric | **Length of last contact (seconds).** Strongly tied to outcome but **not known before calling**. | **Dropped for modeling to avoid target leakage.** Keep in raw data only. |
| campaign | numeric | Number of contacts made during this campaign for the client (includes last contact). | Used as-is; standardized. Often inversely related to success. |
| pdays | numeric | Days since last contact in a previous campaign (999 = never contacted). | Used with care; left numeric and standardized; 999 carries special meaning. |
| previous | numeric | Number of contacts before this campaign for this client. | Used as-is; standardized. |
| poutcome | categorical | Outcome of previous marketing campaign for this client (success, failure, non-existent). | One‑hot encoded; key signal for the simple rule baseline. |
| emp.var.rate | numeric | Employment variation rate (quarterly macro indicator). | Used as-is; standardized. |
| cons.price.idx | numeric | Consumer price index (monthly). | Used as-is; standardized. |
| cons.conf.idx | numeric | Consumer confidence index (monthly). | Used as-is; standardized. |
| euribor3m | numeric | 3‑month Euribor rate (daily). | Used as-is; standardized. |
| nr.employed | numeric | Number of employees (quarterly). | Used as-is; standardized. |
| y | categorical (yes/no) | Outcome label in the source data: whether the client subscribed (yes) or not (no). | **Not used directly**; mapped to `target`. |
| target | binary (0/1) | Engineered field for modeling: 1 if `y == "yes"`, else 0. This is the outcome we predict. | Modeling target. |

## Additional notes
- **Encoding & scaling:** Categorical fields are one‑hot encoded; numeric fields are standardized (variance scaling).  
- **Data split:** Stratified train/test (typically 75/25) to preserve the share of “yes” customers.  
- **Leakage control:** `duration` excluded from all models because it’s only known after we’ve called the customer.  
- **Economics:** Default analysis used €20 contact cost and €250 value per accepted save; these can be changed in the scripts.
