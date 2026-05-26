## Complete AML Intelligence Platform — Final Architecture Specification 

Three independent systems. System 1 (Generator) and System 3 (Evaluation Engine) are fully standalone. System 2 (Detection Platform) is internally organized into Shared - Infrastructure → 4 Detection Layers → Post Detection. Every component specifies libraries, input schema, processing logic, and output schema. 

## MASTER SYSTEM MAP 

╔══════════════════════════════════════════════════════════════════╗ — ║              SYSTEM 1 AML ECOSYSTEM GENERATOR                 ║ ║ ║ ║  G1 Account Population Builder ║ ║  G2 Normal Transaction Generator ║ ║  G3 Laundering Scenario Injector ║ ║  G4 Dataset Splitter & Exporter ║ ║ ║ ║  Outputs: historical_transactions.csv ║ ║ stream_transactions.csv ║ ║           hidden_ground_truth.csv ◄── SEALED                    ║ ╚══════════════════════════════════════════════════════════════════╝ │ │ ▼ ▼ historical_transactions.csv stream_transactions.csv │ ▼ ╔══════════════════════════════════════════════════════════════════╗ — ║            SYSTEM 2 AML DETECTION PLATFORM                    ║ ║ ║ ║ ┌─────────────────────────────────────────────────────────┐ ║ ║ │  SHARED INFRASTRUCTURE                                  │ ║ ║ │  S1 Feature Engineering Pipeline          (Offline) │ ║ ║ │  S2 Directed Multigraph Builder (Offline) │ ║ ║ │  S3 Artifact Persistence                  (Offline) │ ║ - ║ │  S4 Real Time Stream Ingestion (Online) │ ║ ║ │  S5 Dynamic Feature Update               (Online) │ ║ ║ │  S6 Dynamic Multigraph Update             (Online) │ ║ ║ └─────────────────────────────────────────────────────────┘ ║ ║ ║ ║ ┌─────────────────────────────────────────────────────────┐ ║ — - ║ │  LAYER 1 RULE BASED INTELLIGENCE                      │ ║ - ║ │  L1  Rule Based AML Engine                (Online) │ ║ 

║ └─────────────────────────────────────────────────────────┘ ║ ║ ║ ║ ┌─────────────────────────────────────────────────────────┐ ║ — ║ │  LAYER 2 GRAPH ANALYTICS INTELLIGENCE                 │ ║ ║ │  L2A Graph Feature Preprocessor (Offline) │ ║ ║ │  L2B Graph Analytics Engine              (Offline) │ ║ ║ └─────────────────────────────────────────────────────────┘ ║ ║ ║ ║ ┌─────────────────────────────────────────────────────────┐ ║ — ║ │  LAYER 3 BEHAVIORAL ANOMALY INTELLIGENCE              │ ║ ║ │  L3A Behavioral Anomaly Model Training   (Offline) │ ║ ║ │  L3B Behavioral Anomaly Inference        (Online) │ ║ ║ └─────────────────────────────────────────────────────────┘ ║ ║ ║ ║ ┌─────────────────────────────────────────────────────────┐ ║ — - ║ │  LAYER 4 SELF SUPERVISED GRAPH INTELLIGENCE           │ ║ ║ │  L4A TGN / GNN Training                  (Offline) │ ║ ║ │  L4B TGN Inference Engine                (Online) │ ║ ║ └─────────────────────────────────────────────────────────┘ ║ ║ ║ ║ ┌─────────────────────────────────────────────────────────┐ ║ - ║ │  POST DETECTION                                         │ ║ ║ │  P1  Risk Fusion Engine                  (Online) │ ║ ║ │  P2  Alert Management System (Online) │ ║ ║ │  P3  Frontend Dashboard                  (Online) │ ║ ║ └─────────────────────────────────────────────────────────┘ ║ ╚══════════════════════════════════════════════════════════════════╝ │ ▼  generated_alerts.csv ╔══════════════════════════════════════════════════════════════════╗ — ║             SYSTEM 3 EVALUATION ENGINE                        ║ ║ ║ ║  E1 Alert-Ground Truth Joiner ║ ║  E2 Transaction-Level Evaluator ║ ║  E3 Community-Level Evaluator ║ ║  E4 Pattern Coverage Analyzer ║ ║  E5 Report Generator ║ ╚══════════════════════════════════════════════════════════════════╝ 

## — SYSTEM 1 AML ECOSYSTEM GENERATOR 

Purpose: Generate a realistic synthetic banking ecosystem with fully labeled laundering scenarios embedded across both historical and streaming datasets. Runs once, produces three files, and is then sealed from the Detection Platform. 

Research basis: IBM AMLworld / IT-AML (Altman et al., NeurIPS 2023 D&B); AMLSim typologies (Suzumura & Kanezashi, IBM 2021); Benford's Law digit distributions (Chen & Tsourakakis, KDD 2022). 

## Generator Configuration — config.json (Input to all G-components) 

json 

{ "seed": 42, "num_accounts": 5000, "num_historical_transactions": 500000, "num_stream_transactions": 50000, "fraud_ratio": 0.008, "history_days": 90, "stream_days": 10, "banks": ["SBI","HDFC","ICICI","Axis","PNB","Kotak","YES","BOB"], "countries": ["IN","US","AE","SG","GB","CN","MU","NG","PK","CH"], "high_risk_countries": ["AE","MU","CN","NG","PK"], "transaction_types": ["NEFT","RTGS","IMPS","Wire","UPI","Card"], "payment_channels": ["Mobile","Web","ATM","Branch"], "amount_distribution": { "normal_mean": 15000, "normal_std": 40000, "min": 100, "max": 5000000 }, "structuring_threshold": 1000000, "scenario_probabilities": { "structuring": 0.22, "circular_laundering": 0.15, "layering_chain": 0.18, "fan_in": 0.10, "fan_out": 0.10, "fraud_ring": 0.08, "dormant_activation": 0.05, "velocity_burst": 0.05, "cross_border_layering": 0.04, "round_tripping": 0.03 } } 

## — G1 Account Population Builder 

Purpose: Generate the universe of synthetic bank accounts with realistic demographic and KYC attributes. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|faker|≥19.0|Names,addresses, occupations|



|Library|Version|Purpose|
|---|---|---|
|numpy|≥1.26|Statisticalsampling|
|pandas|≥2.0|DataFrame construction|
|uuid|stdlib|Unique accountIDs|
|random|stdlib|Stochastic behavior|



## Input 

config.json → num_accounts, banks, countries, seed 

## Processing 

Samples customer type distribution (70% Individual, 25% Corporate, 5% NBFC), assigns KYC levels weighted toward level 2, generates realistic account ages, assigns home banks and countries, and flags 3% of accounts as "shell" business types for later scenario injection. 

## — Output accounts.parquet (internal, not exported) 

|Column|Type|Description|
|---|---|---|
|account_id|str|UUID4accountidentifer|
|account_name|str|Faker-generated full name|
|bank|str|Assigned bank|
|home_country|str|ISO3166-2countrycode|
|account_created_date|date|Account opening date|
|kyc_level|int (0–3)|0=none, 1=basic, 2=standard, 3=enhanced|
|occupation_type|str|Salaried/Business /Student /Retired|
|business_type|str|Retail/Import-Export /Services /Shell/NBFC|
|customer_type|str|Individual/Corporate/NBFC|
|is_shell|bool|Designatedshell entity fag|
|initial_balance|float64|Starting accountbalance(INR)|
|typical_tx_amount_mean|float64|Account-level amountbaseline|



|Column|Type|Description|
|---|---|---|
|typical_tx_amount_std|float64|Account-level amount variance|
|typical_tx_frequency_per_day|float32|Normal daily transactionrate|



## — G2 Normal Transaction Generator 

Purpose: Generate the baseline population of legitimate transactions that form the behavioral fabric of the ecosystem. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|numpy|≥1.26|Amount sampling, timestampgeneration|
|pandas|≥2.0|DataFrame construction|
|faker|≥19.0|IP addresses,device IDs, remarks|
|scipy.stats|≥1.11|Benford-law-compliantamount sampling|
|uuid|stdlib|Transaction IDs|



## Input 

accounts.parquet ← from G1 config.json → num_historical_transactions, amount_distribution, history_days 

## Processing 

For each transaction: samples sender/receiver pair weighted by social graph proximity (same bank = 60% probability), samples amount from lognormal calibrated per account's 

typical_tx_amount_mean , assigns timestamp following intraday activity curve (peak 9–11am, 3–5pm), generates device/IP consistent with account history (80% repeated device), computes balance updates, and samples amount leading digit such that the full population follows Benford's law (chi-sq p > 0.05). 

## — Output normal_transactions.parquet (internal) 

|Column|Type|Description|
|---|---|---|
|transaction_id|str (UUID4)|Uniquetransaction ID|
|timestamp|datetime64[ns]|ISO8601 timestamp|
|sender_account|str|SenderaccountID|
|receiver_account|str|ReceiveraccountID|
|sender_bank|str|Sending institution|
|receiver_bank|str|Receiving institution|
|sender_country|str|ISO3166-2|
|receiver_country|str|ISO3166-2|
|amount|float64|Transaction amount (INR)|
|currency|str|ISO4217|
|transaction_type|str|NEFT/RTGS/IMPS/Wire/UPI/Card|
|payment_channel|str|Mobile/Web/ATM/Branch|
|device_id|str|Hashed devicefngerprint|
|ip_address|str|IPv4|
|geo_latitude|float64|Origin latitude|
|geo_longitude|float64|Origin longitude|
|merchant_category|str|MCC label|
|transaction_status|str|Success /Failed/Pending|
|sender_balance_before|float64|Pre-tx senderbalance|
|sender_balance_after|float64|Post-tx senderbalance|
|receiver_balance_before|float64|Pre-tx receiverbalance|
|receiver_balance_after|float64|Post-tx receiverbalance|
|is_international|bool|Cross-border fag|



|Column|Type|Description|
|---|---|---|
|remarks|str|Free-textmemo|
|amount_leading_digit|int (1–9)|First signifcantdigit|
|is_suspicious|bool|Always<br>False here|



## G3 — Laundering Scenario Injector 

Purpose: Inject synthetic laundering scenarios into the transaction population, building fraud rings using NetworkX and recording complete ground-truth metadata. 

Research basis: AMLSim typologies; IBM AMLworld scenario definitions (Altman et al., NeurIPS 2023). 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|networkx|≥3.2|Fraudring graph construction|
|numpy|≥1.26|Stochasticscenario parameters|
|pandas|≥2.0|Transaction injection|
|uuid|stdlib|Ring andscenarioIDs|
|faker|≥19.0|Cover story remarks|



## Input 

normal_transactions.parquet ← from G2 accounts.parquet ← from G1 config.json → fraud_ratio, scenario_probabilities, structuring_threshold 

## Scenario Definitions 

|Scenario|Graph Pattern|KeyParameters|
|---|---|---|
|structuring|LinearA→Brepeated|5–15 sub-thresholdtxns,amount ∈ [850K,|
|||999K]INR|



**==> picture [529 x 377] intentionally omitted <==**

**----- Start of picture text -----**<br>
Scenario Graph Pattern Key Parameters<br>–<br>circular_laundering Directed cycle Ring size 3 6, full cycle within 48h<br>A→B→C→A<br>layering_chain Linear chain 3–8 hops, each hop changes bank+country<br>A→B→C→…→Z<br>fan_in Star with target at center 5–15 sources → 1 aggregator<br>fan_out Star with source at 1 distributor → 5–15 targets<br>center<br>fraud_ring Dense clique 4–10 accounts, edge density > 0.6<br>dormant_activation Single node burst 0 txns for ≥180d, then ≥10 txns in 24h<br>velocity_burst Temporal cluster ≥15 txns within 1h window<br>cross_border_layering Chain via high-risk ≥2 hops through  high_risk_countries<br>countries<br>round_tripping Directed cycle via A(IN)→B(AE)→C(SG)→A(IN)<br>international<br>**----- End of picture text -----**<br>


## — Output 1 suspicious_transactions.parquet (internal) 

Same schema as normal_transactions.parquet with is_suspicious = True and an additional column: 

|Column|Type||Description|
|---|---|---|---|
|fraud_ring_id|str|(UUID4)|Laundering networkthis txbelongs to|
|laundering_stage|str||Placement /Layering/Integration|
|synthetic_pattern_type|str||Scenarioenum|



## — Output 2 ground_truth_records.parquet (internal, feeds G4) 

|Column|Type|||Description|
|---|---|---|---|---|
|transaction_id|str|||Referenceto transaction|
|suspicious_flag|str|||Suspicious /Normal|
|fraud_ring_id|str|/|null|Laundering network ID|



|Column|Type|Description|
|---|---|---|
|laundering_stage|str / null|Placement /Layering/Integration|
|suspicious_cluster_id|str|Ground-truth communitylabel|
|synthetic_pattern_type|str|Scenarioenum|
|scenario_description|str|Human-readable explanation|
|scenario_severity|str|Low /Medium/High/Critical|
|entry_account|str|Initialplacementnode|
|exit_account|str|Final integration node|
|num_hops|int|Layering depth|
|ring_size|int|Accountsinring|
|total_ring_value|float64|Total INR laundered inring|



## — G4 Dataset Splitter & Exporter 

Purpose: Merge normal and suspicious transactions, sort chronologically, split 90/10 into historical and stream sets, and export the three final output files. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|pandas|≥2.0|Merge, sort, split,export|
|numpy|≥1.26|Shufing|



## Input 

normal_transactions.parquet ← from G2 suspicious_transactions.parquet ← from G3 ground_truth_records.parquet ← from G3 

## Processing 

Concatenate all transactions, sort by timestamp, take first 90% of chronological records as historical and remaining 10% as stream (ensuring laundering scenarios span both sets). Export 

three files. 

## — Output Final Exported Files 

— historical_transactions.csv Full schema (all columns from G2 output, minus 

is_suspicious , minus fraud_ring_id , minus laundering_stage ). 450K rows approx. 

— stream_transactions.csv Identical schema to historical_transactions.csv . 50K rows - - approx. Streamed one by one by System 2. 

— hidden_ground_truth.csv Full schema from ground_truth_records.parquet . Never 

passed to System 2. Contains rows for all transactions (Normal rows have nulls in fraud-specific columns). 

## — SYSTEM 2 AML DETECTION PLATFORM 

## SHARED INFRASTRUCTURE 

Components that prepare data structures consumed by multiple detection layers. No scoring or detection logic resides here. 

## — S1 Feature Engineering Pipeline 

Offline | Feeds: Layer 1, Layer 3 

- - Purpose: Transform raw historical transactions into per account and per transaction behavioral — feature vectors. Pure computation no thresholds, no anomaly decisions. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|pandas|≥2.0|Rollingwindowaggregations,groupby|
|numpy|≥1.26|Statistical computation|
|scikit-learn|≥1.3|StandardScaler|
|scipy.stats|≥1.11|Entropy,chi-square Benfordtest|



|Library|Version|Purpose|
|---|---|---|
|geopy.distance|≥2.4|Haversine geo-distance|
|datetime|stdlib|Temporal gapcomputation|



## Input 

historical_transactions.csv Schema: all columns as defined in G4 output 

## Processing 

Groups transactions by sender_account . For each account computes rolling window statistics over 1h, 6h, 24h, 7d windows. Applies Haversine between consecutive transaction geolocations. Computes Shannon entropy over receiver distribution. Runs Benford chi-square test per account over 30-day amount windows. Fits and saves StandardScaler over full feature matrix. 

— Output behavioral_features.parquet 

## Temporal Features 

|Column|Type|Description|
|---|---|---|
|transaction_id|str|Rowkey|
|account_id|str|Account (sender perspective)|
|tx_velocity_1h|float32|Txcountin last 1hour|
|tx_velocity_6h|float32|Txcountin last 6hours|
|tx_velocity_24h|float32|Txcountin last 24hours|
|tx_velocity_7d|float32|Txcountin last 7days|
|avg_amount_7d|float64|7-day rolling mean amount|
|std_amount_7d|float64|7-day rollingstd amount|
|tx_gap_seconds|float64|Seconds sinceprior transaction|
|night_tx_ratio|float32|Ratio of22:00–06:00 transactions (30d)|
|weekend_tx_ratio|float32|Ratio of Sat/Suntransactions (30d)|
|hour_of_day|int8|Hour ofthis transaction(0–23)|
|day_of_week|int8|Day (0=Mon… 6=Sun)|



## Behavioral Features 

|Column|Type|Description|
|---|---|---|
|beneficiary_count_7d|int32|Uniquereceiversin7days|
|beneficiary_count_30d|int32|Uniquereceiversin30days|
|receiver_entropy|float32|Shannon entropy ofreceiverdistribution|
|amount_zscore|float32|Z-scorevsaccount's own30d mean|
|avg_daily_volume_30d|float64|Mean daily outgoingvolume|
|round_amount_flag|bool|Amountismultipleof100K/500K/1M|
|sub_threshold_flag|bool|Amount ∈ [850K, 999K]INR|
|amount_leading_digit|int8|First signifcantdigit|
|benford_chi2_score|float32|Chi-sqdeviation from Benford distribution|



## Geographic Features 

|Column|Type|Description|
|---|---|---|
|geo_distance_km|float32|Haversine distance from last transaction|
|country_switch_count_7d|int16|Sendercountrychangesin7days|
|impossible_travel_flag|bool|Impliedspeed> 900km/h|
|high_risk_country_flag|bool|Either partyin high-risk jurisdiction|
|cross_border_ratio_30d|float32|Internationaltx ratio over 30days|



## Device Features 

|Column|Type|Description|
|---|---|---|
|device_change_count_7d|int16|Deviceswitchesin7days|
|new_device_flag|bool|Deviceunseen before for thisaccount|
|shared_device_count|int16|Otheraccounts usingthisdevice|
|ip_change_count_24h|int16|DistinctIPs used in24hours|



— Also outputs: feature_scaler.pkl fitted StandardScaler over full feature matrix, shape (N, 45) . 

## — S2 Directed Multigraph Builder 

Offline | Feeds: Layer 2, Layer 4 

Purpose: Construct the foundational nx.MultiDiGraph where every transaction is preserved as a unique directed edge. This is the data structure consumed by both graph layers. No scoring logic. 

Research basis: Egressy et al., Multi-GNN, AAAI 2024 — multigraph expressivity proof showing standard simple-graph GNNs cannot distinguish structuring on multigraphs. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|networkx|≥3.2|nx.MultiDiGraph construction|
|pandas|≥2.0|Edge attribute loading|
|joblib|≥1.3|Graphserialization|
|neo4j|≥5.0 (optional)|Production graphpersistence|



## Input 

historical_transactions.csv 

Schema: transaction_id, timestamp, sender_account, receiver_account, amount, currency, transaction_type, payment_channel, sender_country, receiver_country, is_international, sender_bank, receiver_bank, kyc_level, device_id 

## Processing 

Creates nx.MultiDiGraph() . For each transaction row: adds sender and receiver as nodes (if not existing) with account-level attributes; adds a directed edge (sender → receiver, key=transaction_id) with all transaction-level attributes. Does NOT collapse multiple — transactions between the same pair each is a separate edge (multigraph). 

## Output 

transaction_multigraph.pkl    nx.MultiDiGraph 

Nodes: account_id Node attrs: bank, home_country, kyc_level, customer_type, account_created_date Edges: one per transaction (multigraph) Edge attrs: transaction_id, timestamp, amount, currency, transaction_type, payment_channel, is_international, sender_country, receiver_country node_index.json { account_id (str) → node_index (int) } edge_index.json { transaction_id (str) → (u, v, edge_key) } 

## — S3 Artifact Persistence 

Offline | Feeds: All online components 

- Purpose: Package all offline produced artifacts into a structured artifact store. The online - pipeline cold starts by loading from this store. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|joblib|≥1.3|.pkl  serialization|
|torch|≥2.1|.pt modelsaving|
|numpy|≥1.26|.npy embeddingsaving|
|pathlib|stdlib|Directorymanagement|
|boto3|≥1.34 (optional)|S3 upload|



## Input 

All outputs from S1, S2, L2A, L2B, L3A, L4A. 

## Output — artifacts/ directory 

artifacts/ `├` ── transaction_multigraph.pkl        ← from S2 `├` ── node_index.json ← from S2 `├` ── edge_index.json ← from S2 `├` ── behavioral_features.parquet ← from S1 `├` ── feature_scaler.pkl                ← from S1 

`├` ── graph_features.parquet ← from L2A `├` ── community_profiles.parquet ← from L2B `├` ── suspicious_paths.parquet ← from L2B `├` ── behavioral_profiles.parquet ← from L3A `├` ── isolation_forest.pkl              ← from L3A `├` ── lof_model.pkl                     ← from L3A `├` ── autoencoder.pt ← from L3A `├` ── tgn_model.pt ← from L4A `├` ── node_embeddings.npy ← from L4A  shape (N, 64) `├` ── node_embedding_index.json ← from L4A └── tgn_memory_state.pkl              ← from L4A 

## — - S4 Real Time Stream Ingestion 

## Online | Feeds: All online layers 

Purpose: Read stream_transactions.csv and emit one validated TransactionEvent per cycle to all downstream components. Acts as the system event bus — no detection logic. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|fastapi|≥0.110|REST+WebSocket server|
|asyncio|stdlib|Async eventloop|
|websockets|≥12.0|Livepushtodashboard|
|pandas|≥2.0|CSVread androwiteration|
|pydantic|≥2.0|Event schemavalidation|



## Input 

stream_transactions.csv 

Schema: identical to historical_transactions.csv 

## — Output TransactionEvent (Pydantic model, emitted once per transaction) 

python 

class TransactionEvent(BaseModel): transaction_id: str timestamp: datetime sender_account: str receiver_account: str sender_name: str receiver_name: str sender_bank: str receiver_bank: str sender_country: str receiver_country: str amount: float currency: str transaction_type: str # NEFT/RTGS/IMPS/Wire/UPI/Card payment_channel: str # Mobile/Web/ATM/Branch device_id: str ip_address: str geo_latitude: float geo_longitude: float merchant_category: str transaction_status: str # Success/Failed/Pending sender_balance_before: float sender_balance_after: float receiver_balance_before: float receiver_balance_after: float kyc_level: int # 0–3 is_international: bool remarks: str amount_leading_digit: int # 1–9 

## — S5 Dynamic Feature Update 

Online | Feeds: Layer 1 (L1), Layer 3 (L3B) 

- Purpose: Incrementally update per account behavioral features using the latest TransactionEvent . Maintains a rolling state window in memory. Produces the LiveFeatureVector consumed by Layer 1 and Layer 3. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|pandas|≥2.0|Rollingwindow update|



|Library|Version|Purpose|
|---|---|---|
|numpy|≥1.26|Fastaggregation|
|geopy.distance|≥2.4|Haversine computation|
|scipy.stats|≥1.11|Live Benford chi-square|
|redis|≥5.0 (optional)|In-memoryaccount state cache|



## Input 

TransactionEvent ← from S4 behavioral_features.parquet ← loaded from S3 at startup (historical baselines) feature_scaler.pkl                    ← loaded from S3 at startup 

## Processing 

Looks up account's historical baseline from behavioral_profiles.parquet . Appends new transaction to rolling window. Recomputes all velocity, entropy, geo, device, and Benford features over updated window. Applies StandardScaler transform. 

## — Output LiveFeatureVector 

python 

class LiveFeatureVector(BaseModel): transaction_id: str sender_account: str # All 45 feature columns matching behavioral_features.parquet schema tx_velocity_1h: float tx_velocity_6h: float tx_velocity_24h: float tx_velocity_7d: float avg_amount_7d: float std_amount_7d: float tx_gap_seconds: float night_tx_ratio: float weekend_tx_ratio: float hour_of_day: int day_of_week: int beneficiary_count_7d: int beneficiary_count_30d: int receiver_entropy: float amount_zscore: float avg_daily_volume_30d: float round_amount_flag: bool sub_threshold_flag: bool amount_leading_digit: int benford_chi2_score: float geo_distance_km: float country_switch_count_7d: int impossible_travel_flag: bool high_risk_country_flag: bool cross_border_ratio_30d: float device_change_count_7d: int new_device_flag: bool shared_device_count: int ip_change_count_24h: int scaled_feature_vector: list[float] # StandardScaler output, shape (45,) 

## — S6 Dynamic Multigraph Update 

Online | Feeds: Layer 2 (graph_score), Layer 4 (L4B) 

Purpose: Add the new transaction as a directed edge to the live nx.MultiDiGraph . Incrementally recompute k-hop subgraph features for affected sender and receiver nodes. Detect whether the new edge closes a cycle. Produces LiveGraphFeatureVector consumed by Layer 2 scoring and Layer 4 inference. 

Research basis: BRIGHT (Lu et al., CIKM 2022) — k-hop on-demand neighborhood fetch for sub-100ms latency. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|networkx|≥3.2|add_edge  on<br>MultiDiGraph ,cycle detection|
|numpy|≥1.26|Local metricrecomputation|
|infomap|≥1.7|Local Infomap re-runon afectedsubgraph|
|scipy.sparse|≥1.11|Efcientadjacencyfork-hopfetch|



## Input 

TransactionEvent ← from S4 transaction_multigraph (live) ← loaded from S3, mutated in-memory graph_features.parquet (baseline) ← loaded from S3 at startup community_profiles.parquet ← loaded from S3 at startup 

## Processing 

1. Calls G.add_edge(sender, receiver, key=transaction_id, **edge_attrs) . 

- 

- 2. Extracts 2 hop directed neighborhood of sender using BFS on MultiDiGraph . 

3. Recomputes local graph metrics (degree, cycle count, fan scores) for sender and receiver. 

4. Checks if new edge (sender→receiver) closes any existing directed path receiver→...→sender (simple cycle detection via DFS on 3-hop subgraph). 

5. Fetches sender's community profile from community_profiles.parquet . 

## — Output LiveGraphFeatureVector 

python 

class LiveGraphFeatureVector(BaseModel): transaction_id: str sender_account: str receiver_account: str sender_in_degree: int sender_out_degree: int sender_in_degree_unique: int sender_out_degree_unique: int sender_2hop_cycle_count: int sender_3hop_cycle_count: int sender_fan_in_score: float sender_fan_out_score: float sender_pagerank: float sender_betweenness: float sender_community_id: int sender_community_size: int sender_community_density: float sender_community_risk_score: float # from community_profiles baseline sender_benford_chi2_community: float receiver_in_degree: int receiver_out_degree: int receiver_community_id: int receiver_community_risk_score: float edge_creates_cycle: bool # NEW: does this edge close a cycle? cycle_length: int # hop count of closed cycle (0 if none shared_community: bool # sender + receiver in same community two_hop_neighborhood: list[str] # account IDs in 2-hop neighborhood two_hop_edge_list: list[tuple] # (u, v, timestamp, amount) per edge   

## — - LAYER 1 RULE BASED INTELLIGENCE 

Nature: Deterministic. Hardcoded typology rules. No training phase. Offline component: None. Online component: L1 — Rule-Based AML Engine. 

## — - L1 Rule Based AML Engine 

## Online 

Purpose: Execute 13 deterministic AML rules against the live feature vectors. Produces a normalized rule_score and human-readable explanations for every triggered rule. 

Research basis: FATF typology codification; NICE Actimize SAM / SAS AML production rule design; Tookitaki structuring diagnosis. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|pandas|≥2.0|Rule evaluationon feature frames|
|numpy|≥1.26|Threshold logic andscore aggregation|



## Input 

python LiveFeatureVector ← from S5 LiveGraphFeatureVector ← from S6 TransactionEvent ← from S4 

## Rules 

|Rule|Name|Condition|Condition|Condition||||Weight|
|---|---|---|---|---|---|---|---|---|
|ID|||||||||
|R01|High-valuetransfer|amount||> 1_000_000||||1.0|
|R02|Structuring|sub_threshold_flag ANDtx_velocity_1h≥ 3||||||1.0|
|R03|Velocity spike|tx_velocity_1h>||||10ORtx_velocity_24h>||0.8|
|||50|||||||
|R04|Dormantactivation|tx_gap_seconds >||||15_552_000|AND amount >|0.9|
|||100_000|||||||
|R05|Impossibletravel|impossible_travel_flag==True||||||0.9|
|R06|High-risk jurisdiction|high_risk_country_flag AND||||||0.7|
|||is_international|||||||
|R07|Device anomaly|new_device_flag|||AND amount > 200_000|||0.6|
|R08|Shared device|shared_device_count > 5||||||0.7|
|R09|Excessive benefciaries|beneficiary_count_7d> 20||||||0.8|
|R10|Cycle closure|edge_creates_cycle==True||||||1.0|



Condition 

Weight 

Rule Name ID 

|R11|Round-amount|round_amount_flag ANDtx_velocity_24h> 5|0.6|
|---|---|---|---|
||structuring|||
|R12|KYC mismatch|kyc_level== 0AND amount > 500_000|0.8|
|R13|Benford anomaly|benford_chi2_score> 3.84  (chi-sqcritical at|0.7|
|||α=0.05)||



## Processing 

For each rule Ri with weight wi: fire_i = evaluate_condition(Ri) → 0 or 1 raw_score = `Σ` (fire_i × wi) rule_score = min(100, (raw_score / max_possible_raw) × 100) 

## — Output RuleEngineOutput 

## python 

class RuleEngineOutput(BaseModel): transaction_id: str rule_score: float # 0–100, normalized triggered_rules: list[str] # e.g. ["R02", "R10"] rule_explanations: list[str] # human-readable per rule rule_count: int # number of fired rules 

## — LAYER 2 GRAPH ANALYTICS INTELLIGENCE 

Nature: Topological graph algorithms. No ML training. Computes structural risk from the transaction multigraph. Offline components: L2A (Graph Feature Preprocessor), L2B (Graph Analytics Engine). Online exposure: 

LiveGraphFeatureVector.sender_community_risk_score from S6 becomes graph_score in P1 (Risk Fusion). 

## — L2A Graph Feature Preprocessor (GFP) 

Offline 

- - Purpose: Compute per node subgraph pattern features over the full historical multigraph. Captures multi-hop laundering structures invisible to per-transaction features. 

Research basis: Blanuša et al., Graph Feature Preprocessor, ICAIF 2024 (IBM Research) — GFPfed XGBoost matches GNN performance with greater interpretability; Chen & Tsourakakis, AntiBenford Subgraphs, KDD 2022 — Benford-law dense subgraph anomaly detection. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|networkx|≥3.2|Degree,PageRank,betweenness,cycle detection|
|numpy|≥1.26|Feature aggregation|
|pandas|≥2.0|Feature DataFrame|
|scipy.sparse|≥1.11|Efcientadjacencyfor subgraph mining|
|scipy.stats|≥1.11|Benford chi-squarepercommunity|
|infomap|≥1.7|Directed communitydetection|



## Input 

transaction_multigraph.pkl    ← from S2 Schema: nx.MultiDiGraph with node and edge attributes as defined in S2 

## Processing 

1. Computes in_degree , out_degree for every node on the MultiDiGraph (counting multiedges separately — not unique neighbors). 

2. Computes in_degree_unique , out_degree_unique (unique neighbors only). 

3. Runs PageRank with damping 0.85 on the condensed weighted digraph. 

4. Runs betweenness centrality on condensed digraph (normalized). 

5. Runs local clustering coefficient on undirected projection. 

- - 

- 6. For each node: counts directed 2 hop cycles (A→B→A) and 3 hop cycles (A→B→C→A) using DFS over MultiDiGraph . 

7. Computes fan_in_score = in_degree_unique / (in_degree_unique + out_degree_unique + `ε` ) . 

8. Computes fan_out_score = out_degree_unique / (in_degree_unique + out_degree_unique + `ε` ) . 

9. Computes scatter_gather_score = presence of fan-in followed by fan-out within 2 hops. 

10. Runs directed Infomap to assign community_id to each node. 

## 11. Per community: computes density, total flow value, dominant_pattern , has_cycle , max_cycle_length . 

12. Per community: runs Benford chi-square test on all transaction amounts. 

13. Computes ego-level inbound/outbound volumes over 7-day window. 

## — Output graph_features.parquet 

|Column|Type|Description|
|---|---|---|
|account_id|str|Node identifer|
|in_degree|int32|Incoming edge count (multi-edgescounted)|
|out_degree|int32|Outgoing edge count (multi-edgescounted)|
|in_degree_unique|int32|Uniquesendercount|
|out_degree_unique|int32|Uniquereceivercount|
|pagerank_score|float32|PageRank(damping=0.85)|
|betweenness_centrality|float32|Normalized betweenness|
|clustering_coefficient|float32|Local clustering(undirectedprojection)|
|2hop_cycle_count|int32|A→B→A cycles|
|3hop_cycle_count|int32|A→B→C→A cycles|
|fan_in_score|float32|Aggregationtendency (0–1)|
|fan_out_score|float32|Distributiontendency (0–1)|
|scatter_gather_score|float32|Fan-inthen fan-out within2hops|
|community_id|int32|Directed Infomaplabel|
|community_size|int32|Membersin community|
|community_density|float32|Internal edge density|
|ego_in_volume_7d|float64|Total incoming INR in7days|
|ego_out_volume_7d|float64|Totaloutgoing INR in7days|
|volume_asymmetry|float32|(out −in) / (out +in)|
|benford_chi2_community|float32|Chi-sqBenford deviationof community'samounts|



## — L2B Graph Analytics Engine 

## Offline 

- Purpose: Run community detection, classify communities by dominant pattern, trace multi hop - - fund flow paths, score each community for group level risk. 

Research basis: Bellei et al., Elliptic2, KDD 2024 (subgraph-level classification); Cheng et al., - - Group Aware Deep Graph Learning, IEEE TKDE 2023 (group level scoring); directed Infomap (replacing undirected Louvain). 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|networkx|≥3.2|SCC, pathfnding,DFS|
|infomap|≥1.7|Directed communitydetection|
|scipy.sparse|≥1.11|Dense-subgraph greedy peeling|
|pandas|≥2.0|Communityandpath DataFrames|
|numpy|≥1.26|Riskscore aggregation|



## Input 

transaction_multigraph.pkl    ← from S2 graph_features.parquet ← from L2A 

## Processing 

1. Runs directed Infomap on the full MultiDiGraph to assign community membership. 

- 

- 2. For each community: detects dominant_pattern by rule if has_cycle → circular ; elif fan_in_score > 0.7 → fan_in ; elif fan_out_score > 0.7 → fan_out ; elif chain 

- structure → layering_chain ; else mixed . 

3. Computes community_risk_score as weighted combination of density, cycle presence, Benford anomaly, and community size. 

- 

- 4. Runs DFS to enumerate all simple directed paths of length 2 8 hops for communities flagged as layering_chain or circular . 

- - 

- 5. Runs greedy dense subgraph peeling to surface Benford anomalous dense subgraphs. 

Output 

## community_profiles.parquet 

|Column|Type|Description|
|---|---|---|
|community_id|int32|Infomaplabel|
|member_accounts|list[str]|AccountIDs|
|community_size|int32|Node count|
|community_density|float32|Internal edge density|
|total_flow_value|float64|Total INRthrough community|
|dominant_pattern|str|circular /fan_in/fan_out /layering_chain/mixed|
|has_cycle|bool|Containsdirected cycle|
|max_cycle_length|int16|Longestdirected cycle|
|benford_anomaly|bool|Community-level Benford deviationsignifcant|
|community_risk_score|float32|Group-levelrisk baseline(0–100)|



## suspicious_paths.parquet 

|Column|Type|Description|
|---|---|---|
|path_id|str (UUID4)|Path identifer|
|path_nodes|list[str]|Ordered accountIDs|
|path_edges|list[str]|Orderedtransaction IDs|
|path_length|int16|Hopcount|
|total_value|float64|INRfowingthroughpath|
|pattern_type|str|layering_chain/ round_trip / scatter_gather|



## — LAYER 3 BEHAVIORAL ANOMALY INTELLIGENCE 

Nature: Unsupervised ML. Learns normal behavior from history. No labels used. Offline component: L3A — Behavioral Anomaly Model Training. Online component: L3B — Behavioral Anomaly Inference. 

## — L3A Behavioral Anomaly Model Training 

## Offline 

Purpose: Train three complementary unsupervised anomaly models on the full historical feature matrix. Each model captures a different type of deviation from normal. 

Research basis: Standard production deployment (Isolation Forest at HSBC, eBay BRIGHT); — DiGA (Li et al., KDD 2023) extreme class imbalance handling; LaundroGraph (Cardoso et al., ICAIF 2022) — subgraph features improve behavioral model AUC by 12 p.p. 

## Libraries 

**==> picture [529 x 171] intentionally omitted <==**

**----- Start of picture text -----**<br>
Library Version Purpose<br>scikit-learn ≥1.3 IsolationForest , LocalOutlierFactor<br>torch ≥2.1 Autoencoder (encoder-decoder)<br>joblib ≥1.3 Model serialization<br>numpy ≥1.26 Array operations<br>pandas ≥2.0 Feature loading<br>**----- End of picture text -----**<br>


## Input 

behavioral_features.parquet ← from S1 

Columns used: all 45 feature columns 

graph_features.parquet ← from L2A 

Columns merged in: 2hop_cycle_count, 3hop_cycle_count, fan_in_score, fan_out_score, scatter_gather_score, 

community_density, benford_chi2_community 

Merged on: account_id 

Combined shape: (num_accounts, ~52) Scaled via: feature_scaler.pkl from S1 

— Processing Three Models 

## — Model 1 Isolation Forest 

IsolationForest( n_estimators = 200, contamination = 0.01, max_features = 1.0, random_state = 42 ) 

Detects: global statistical outliers 

## — Model 2 Local Outlier Factor 

LocalOutlierFactor( n_neighbors = 30, novelty = True, contamination = 0.01, metric = 'euclidean' ) Detects: local neighborhood density anomalies 

## — Model 3 Autoencoder 

Architecture: Input layer : dim=52 Encoder : Linear(52→32) → ReLU → Linear(32→16) Latent : dim=16 Decoder : Linear(16→32) → ReLU → Linear(32→52) Loss: MSE reconstruction error Epochs: 50, LR: 1e-3, Batch: 256 Detects: accounts whose feature pattern cannot be reconstructed = structurally unlike any normal account 

## Computes autoencoder_baseline_recon_error and iso_forest_baseline_score per account 

for use in online normalization. 

## Output 

isolation_forest.pkl            trained IsolationForest lof_model.pkl                   trained LOF (novelty=True) autoencoder.pt trained PyTorch autoencoder checkpoint behavioral_profiles.parquet per-account baseline stats 

behavioral_profiles.parquet schema: 

account_id                    str mean_amount                   float64 std_amount                    float64 p95_amount                    float64 typical_tx_velocity_24h       float32 typical_beneficiary_count     float32 iso_forest_baseline_score     float32 (score on training data) autoencoder_baseline_recon    float32 (MSE on training data) 

## — L3B Behavioral Anomaly Inference 

## Online 

- Purpose: Score each incoming transaction against all three pre trained anomaly models. - Identify the top 3 driving features. Produce a normalized ensemble anomaly score. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|scikit-learn|≥1.3|IF and LOF<br>.predict()  /<br>.score_samples()|
|torch|≥2.1|Autoencoderforwardpass|
|numpy|≥1.26|Score normalization and aggregation|



## Input 

LiveFeatureVector.scaled_feature_vector ← from S5, shape (52,) with GFP features appended 

isolation_forest.pkl                      ← loaded from S3 at startup lof_model.pkl                             ← loaded from S3 at startup autoencoder.pt ← loaded from S3 at startup 

behavioral_profiles.parquet ← loaded from S3 at startup (per-account baselines) 

## Processing 

iso_raw = isolation_forest.score_samples([x]) → negative; higher = more normal lof_raw = lof_model.score_samples([x]) → negative; higher = more normal ae_mse    = MSE(autoencoder(x), x) → higher = more anomalous 

Normalize each to 0–100 using per-account baseline from behavioral_profiles: iso_score  = normalize(iso_raw) lof_score  = normalize(lof_raw) ae_score   = normalize(ae_mse) 

ensemble_anomaly_score = mean(iso_score, lof_score, ae_score) 

Top-3 anomaly drivers = features with largest |deviation from account baseline| 

## — Output BehavioralAnomalyOutput 

python class BehavioralAnomalyOutput(BaseModel): transaction_id: str iso_forest_score: float # 0–100, higher = more anomalous lof_score: float # 0–100 autoencoder_recon_error: float # raw MSE autoencoder_score: float # 0–100 ensemble_anomaly_score: float # mean of three, 0–100 anomaly_drivers: list[str] # top-3 feature names driving anomaly 

## — - LAYER 4 SELF SUPERVISED GRAPH INTELLIGENCE 

Nature: Deep learning on directed multigraph. Self-supervised with link-prediction objective. Online memory updates per transaction. Offline component: L4A — TGN / GNN Training. Online component: L4B — TGN Inference Engine. 

## — L4A TGN / GNN Training 

## Offline 

- Purpose: Learn normal graph connectivity and temporal structural patterns through self supervised link prediction. No labels used. Produces node embeddings and initialized memory states for all accounts. 

Research basis: Rossi et al., TGN, ICML 2020 Workshop (temporal memory + continuous-time node state); Cardoso et al., LaundroGraph, ICAIF 2022 (link-prediction objective on bipartite graph — 12 p.p. AUC improvement); Egressy et al., Multi-GNN, AAAI 2024 (ego-IDs + port numbering for multigraph expressivity); Bicer et al., MEGA-GNN, arXiv 2412.00241 (bidirectional multi-edge aggregation — 13.31% minority-class F1 improvement). 

## Libraries 

**==> picture [529 x 202] intentionally omitted <==**

**----- Start of picture text -----**<br>
Library Version Purpose<br>torch ≥2.1 Deep learning backend<br>torch-geometric ≥2.4 GNN layers, TemporalData , NeighborLoader<br>torch-geometric-temporal ≥0.54 EvolveGCN , TGN memory utilities<br>numpy ≥1.26 Embedding export<br>joblib ≥1.3 Memory state serialization<br>pandas ≥2.0 Edge list construction<br>**----- End of picture text -----**<br>


## Input 

transaction_multigraph.pkl         ← from S2 

Used as: temporal edge stream sorted by timestamp 

graph_features.parquet ← from L2A 

Used as: initial node feature matrix, shape (N, 20) 

Columns: pagerank_score, betweenness_centrality, fan_in_score, 

fan_out_score, community_density, benford_chi2_community, 

in_degree, out_degree, 2hop_cycle_count, 3hop_cycle_count, 

+ 10 node-level account attributes (kyc_level, customer_type enc, etc.) 

node_index.json ← from S2 

## Model Architecture 

## — - Stage 1 Multigraph Port Numbering (Multi GNN preprocessing) 

## For each node u: 

Sort outgoing edges by timestamp 

Assign port_number ∈ {0, 1, ..., out_degree-1} to each edge Add ego_id = hash(account_id) % embedding_dim as node feature 

- Purpose: makes multi edges (structuring patterns) distinguishable to GNN 

## — Stage 2 TGN Memory Module 

memory:         Dict[node_id → Tensor(memory_dim=64)] (initialized to zeros) time_encoder:   Linear(1 → time_dim=16) (encodes Δt) 

message_fn:     MLP([src_memory | dst_memory | time_enc | edge_features] → msg_dim=64) 

memory_updater: GRU(input=message, hidden=current_memory → updated_memory) 

## Stage 3 — Graph Attention Encoder (MEGA-GNN bidirectional aggregation) 

forward_agg  = mean(messages from outgoing neighbors, direction=→) backward_agg = mean(messages from incoming neighbors, direction=←) node_emb = MLP(concat(forward_agg, backward_agg, ego_id, memory[u])) → Tensor(embedding_dim=64) 

## — - Stage 4 Self Supervised Link Prediction Objective 

Positive pairs:  actual (sender, receiver) transaction pairs from history Negative pairs: random account pairs with no transaction between them (10× ratio) Scorer:          dot_product(emb[u], emb[v]) → logit 

Loss:            BinaryCrossEntropy(logit, label) 

Training forces model to encode: "normal transaction relationships" Anomaly at inference = sender+receiver pair scores low under this model 

## Training config 

epochs: 50 

batch_size: 512 (transaction events per batch) learning_rate: 1e-3 

optimizer:      Adam scheduler:      CosineAnnealingLR 

## Output 

tgn_model.pt                   full trained TGN checkpoint (all weights) 

node_embeddings.npy shape (num_accounts, 64) — post-training baseline embeddings 

node_embedding_index.json { account_id (str) → row_index (int) } 

tgn_memory_state.pkl           { account_id (str) → memory_tensor (shape: 64,) } (initialized from final training state) 

— L4B TGN Inference Engine 

Online 

Purpose: For each incoming transaction: update TGN memory for sender and receiver, run GNN - forward pass on their updated 2 hop neighborhood, compute embedding drift from baseline, and produce a structural anomaly score. 

Research basis: Rossi et al., TGN (memory update mechanism); Di Gennaro et al., Amatriciana, IEEE ICDMW 2024 (temporal GNN: F1=0.76, 55% FP reduction on AMLworld). 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|torch|≥2.1|TGN forwardpass|
|torch-geometric|≥2.4|Subgraph neighbor sampling|
|numpy|≥1.26|Cosine distance,embedding comparison|



## Input 

TransactionEvent ← from S4 

- Used: sender_account, receiver_account, timestamp, amount, transaction_type, is_international 

LiveGraphFeatureVector.two_hop_edge_list ← from S6 

Used: 2-hop directed neighborhood as edge stream for GNN forward pass 

tgn_model.pt ← loaded from S3 at startup node_embeddings.npy (baseline) ← loaded from S3 at startup node_embedding_index.json ← loaded from S3 at startup tgn_memory_state (live) ← loaded from S3, mutated in-memory 

## — - Processing Per Transaction Inference 

Step 1  Build edge feature vector: edge_feat = concat( 

- time_encoder(current_timestamp − last_interaction_timestamp), [amount_normalized, tx_type_onehot, is_international] 

- ) → shape (time_dim + edge_feat_dim,) 

Step 2  Compute messages for sender and receiver: msg_sender = message_fn(memory[sender], memory[receiver], edge_feat) msg_receiver = message_fn(memory[receiver], memory[sender], edge_feat) 

Step 3  Update memory (GRU): 

memory[sender] ← GRU(msg_sender, memory[sender]) 

memory[receiver] ← GRU(msg_receiver, memory[receiver]) 

Step 4  Build 2-hop port-numbered neighborhood from 

LiveGraphFeatureVector.two_hop_edge_list 

Step 5  GNN forward pass (MEGA-GNN bidirectional aggregation): 

new_emb_sender = GraphAttn(memory[sender], 2hop_neighborhood) 

new_emb_receiver = GraphAttn(memory[receiver], 2hop_neighborhood) 

Step 6  Compute embedding drift: 

drift_sender = 1 − cosine_similarity(new_emb_sender, 

baseline_emb[sender]) 

drift_receiver = 1 − cosine_similarity(new_emb_receiver, 

baseline_emb[receiver]) 

Step 7  Compute link anomaly score: 

link_score = 1 − sigmoid(dot(new_emb_sender, new_emb_receiver)) 

(low probability under learned normal-relationship model = suspicious) 

Step 8  Aggregate: 

gnn_anomaly_score = normalize( 

0.4 × drift_sender + 0.3 × drift_receiver + 0.3 × link_score ) × 100 

## — Output GNNInferenceOutput 

## python 

class GNNInferenceOutput(BaseModel): transaction_id: str sender_embedding: list[float] # shape (64,) receiver_embedding: list[float] # shape (64,) sender_embedding_drift: float # cosine distance from baseline, 0–1 receiver_embedding_drift: float # cosine distance from baseline, 0–1 − link_anomaly_score: float # 1 link prediction probability, 0– gnn_anomaly_score: float # aggregated, 0–100 structural_anomaly_explanations: list[str] # e.g. ["cycle closure", "embedding d community_anomaly_flag: bool # community Benford or density anomal 

 

 

## - POST DETECTION COMPONENTS 

— Run after all four layers produce scores. Aggregate, persist, visualize no detection logic. 

## — P1 Risk Fusion Engine 

## Online 

Purpose: Combine the four layer outputs into a transaction_risk_score and a 

- - - group_risk_score (community level). Supports both static weight and learned meta fusion modes. 

Research basis: NVIDIA GNN+XGBoost blueprint (2024); Cheng et al., Group-Aware Deep Graph Learning, IEEE TKDE 2023. 

## Libraries 

**==> picture [541 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
Library Version Purpose<br>numpy ≥1.26 Weighted scoring<br>xgboost ≥2.0 Meta-learner (Phase 2)<br>Inputput<br>python<br>RuleEngineOutput ← from L1 rule_score (0–100)<br>LiveGraphFeatureVector ← from S6 sender_community_risk_score (0–100) → graph_s<br>BehavioralAnomalyOutput ← from L3B    ensemble_anomaly_score (0–100) → anomaly_scor<br>GNNInferenceOutput ← from L4B    gnn_anomaly_score (0–100) → gnn_score<br>TransactionEvent ← from S4 transaction_type, customer_type, kyc_level (f<br> <br>**----- End of picture text -----**<br>


## Inputput 

## Fusion Formula 

## — Phase 1 Static weights (no labeled data) 

transaction_risk_score = 0.30 × rule_score + 0.25 × graph_score + 0.25 × anomaly_score + 0.20 × gnn_score 

## — - Phase 2 Learned meta fusion (after SAR feedback accumulates) 

XGBoost meta-learner: 

features: [rule_score, graph_score, anomaly_score, gnn_score, 

transaction_type_enc, customer_type_enc, kyc_level, community_size, sender_embedding_drift, rule_count] 

target: sar_filed (0/1) 

Output:   learned_risk_score (0–100), calibrated probability 

## Group-Level Score (Cheng et al.) 

group_risk_score = 

   - max(transaction_risk_score across all accounts in community) 

- + 0.15 × community_density 

- + 0.10 × (1 if has_cycle else 0) 

- → clipped to [0, 100] 

## Risk Level Thresholds 

**==> picture [529 x 132] intentionally omitted <==**

**----- Start of picture text -----**<br>
||||
|---|---|---|
|Score|Level|
|–|
|0|30|Low|
|31–60|Medium|
|–|
|61|80|High|
|81–100|Critical|

**----- End of picture text -----**<br>


## — Output FusedRiskOutput 

## python 

**==> picture [479 x 182] intentionally omitted <==**

**----- Start of picture text -----**<br>
|||||||
|---|---|---|---|---|---|
|class|FusedRiskOutput(BaseModel):|
|transaction_id:|str|
|sender_account:|str|
|transaction_risk_score:|float|
|group_risk_score:|float|
|risk_level:|str|# Low/Medium/High/Critical|
|risk_level_group:|str|
|score_breakdown:|dict|#|{rule, graph, anomaly, gnn,|weights}|
|triggered_patterns:|list[str]|# consolidated from all layer|outputs|
|explanation:|str|# human-readable summary|sentence|
|fusion_mode:|str|#|"static_weights"|or|"meta_learner"|

**----- End of picture text -----**<br>


— P2 Alert Management System 

## Online 

Purpose: Persist FusedRiskOutput as structured Alert records when risk is High or Critical. Track investigation workflow. Export alert log for Evaluation Engine. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|fastapi|≥0.110|REST API|
|sqlalchemy|≥2.0|ORM|
|asyncpg|≥0.29|Async PostgreSQL driver|
|pydantic|≥2.0|Schemavalidation|



## Input 

python 

FusedRiskOutput ← from P1 Trigger condition: risk_level ∈ {"High", "Critical"} OR risk_level_group ∈ {"High", "Critical"} 

## Output — Alert (persisted to PostgreSQL + generated_alerts.csv ) 

python 

class Alert(BaseModel): alert_id: str # UUID4 transaction_id: str sender_account: str community_id: int transaction_risk_score: float group_risk_score: float risk_level: str risk_level_group: str triggered_patterns: list[str] rule_explanations: list[str] anomaly_drivers: list[str] structural_anomaly_explanations: list[str] score_breakdown: dict explanation: str alert_status: str # Open/Investigating/SAR_Filed/Closed assigned_to: str | None created_at: datetime updated_at: datetime 

## Also exports: generated_alerts.csv 

|Column|Type|
|---|---|
|alert_id|str|
|transaction_id|str|
|sender_account|str|
|community_id|int|
|transaction_risk_score|float|
|group_risk_score|float|
|risk_level|str|
|triggered_patterns|str (JSON-encoded list)|
|alert_status|str|
|created_at|datetime|



— P3 Frontend Dashboard 

## Online 

- Purpose: Analyst facing investigation UI. Displays live risk stream, alert queue, multigraph - visualization, and per alert score breakdowns. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|next.js|≥14|Frontend framework|
|tailwindcss|≥3|Utility styling|
|recharts|≥2|Risktrend charts, score distribution|
|cytoscape.js|≥3.28|Directed multigraphvisualization|
|framer-motion|≥11|UItransitions|
|react-query|≥5|Server state and cache|
|axios|latest|REST API communication|



## Input 

WebSocket stream from S4/P2 ← live TransactionEvent + Alert pushes REST API from P2 ← alert queue, investigation case data REST API from S2/S6 ← live graph data for Cytoscape renderer 

## Pages 

|Page|Content|
|---|---|
|Dashboard|Liveriskscorestream,alertcountsbylevel, pattern distributionpie|
|AlertQueue|Filterabletable: risk_level, pattern,community,alert_status|
|Investigation|Single-alertdeep-dive: score breakdown bars,nodetimeline,explanation|
|Graph|Cytoscape.jsmultigraph— red nodes (suspicious),colored communities,highlighted|
|Explorer|cyclepaths,multi-edge display peraccount pair|
|Analytics|PR-AUCtrend,false-positiverate, pattern detection heatmap|
|Upload|Manual<br>stream_transactions.csv  upload forad-hoc analysis|



## — SYSTEM 3 EVALUATION ENGINE 

Purpose: Benchmark detection quality by comparing generated_alerts.csv against the 

sealed hidden_ground_truth.csv . Completely independent. Never runs during detection. No access to detection model internals. 

Research basis: Altman et al., NeurIPS 2023 D&B — minority-class F1 as primary AML metric; Tide reference datasets (2024) — PR-AUC benchmarks. 

## — - E1 Alert Ground Truth Joiner 

Purpose: Merge alert records and ground truth on transaction_id . Build the binary classification frame needed by all evaluators. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|pandas|≥2.0|Join and alignment|
|numpy|≥1.26|Label arrayconstruction|



## Input 

generated_alerts.csv ← from P2 (System 2) 

Required columns: transaction_id, transaction_risk_score, alert_status, 

community_id 

hidden_ground_truth.csv ← from G4 (System 1) 

Required columns: transaction_id, suspicious_flag, fraud_ring_id, 

synthetic_pattern_type, suspicious_cluster_id 

## Processing 

- Left join all stream transactions against alerts on transaction_id . Assign: 

y_true = 1 if suspicious_flag == "Suspicious" else 0 

y_pred = 1 if alert_status ∈ {"Open","Investigating","SAR_Filed"} else 0 y_score = transaction_risk_score / 100 (for threshold-free metrics) 

## — Output joined_evaluation_frame.parquet 

|Column|Type|
|---|---|
|transaction_id|str|
|y_true|int (0/1)|
|y_pred|int (0/1)|
|y_score|float (0–1)|
|fraud_ring_id|str / null|
|community_id_alert|int|
|suspicious_cluster_id|str / null|
|synthetic_pattern_type|str / null|
|scenario_severity|str / null|
|alert_status|str / null|



## — - E2 Transaction Level Evaluator 

- Purpose: Compute all standard and AML specific classification metrics at the individual transaction level. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|scikit-learn|≥1.3|All classifcation metrics|
|numpy|≥1.26|Thresholdsweep|



## Input 

joined_evaluation_frame.parquet ← from E1 

Columns used: y_true, y_pred, y_score 

## Processing 

Computes all metrics at default threshold (risk_score ≥ 61 → High). Also sweeps thresholds from 0–100 to generate PR curve and ROC curve. 

## — Output transaction_metrics.json 

json { "precision": 0.74, "recall": 0.81, "f1_score_macro": 0.70, "minority_class_f1": 0.77, "pr_auc": 0.83, "roc_auc": 0.91, "false_positive_rate": 0.06, "false_negative_rate": 0.19, "total_alerts_generated": 512, "true_positives": 324, "false_positives": 113, "false_negatives": 76, "true_negatives": 49087 } 

## — - E3 Community Level Evaluator 

Purpose: Assess how well the system detects fraud rings as complete communities, not just individual transactions. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|pandas|≥2.0|Community-level groupby|
|numpy|≥1.26|Matchrate computation|



## Input 

joined_evaluation_frame.parquet ← from E1 

Columns used: fraud_ring_id, community_id_alert, suspicious_cluster_id, y_pred 

## Processing 

Groups by fraud_ring_id . A ring is "detected" if ≥ 50% of its transactions were alerted. 

Computes community ID overlap between suspicious_cluster_id and community_id_alert . 

## — Output community_metrics.json 

json 

{ "fraud_ring_detection_rate": 0.72, "partial_ring_detection_rate": 0.85, "community_id_match_score": 0.68, "group_risk_precision": 0.79, "mean_ring_coverage": 0.74 } 

## — E4 Pattern Coverage Analyzer 

- - Purpose: Compute per scenario detection rates, per rule precision, Benford rule audit, and alert latency. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|pandas|≥2.0|Pattern-level groupby|
|numpy|≥1.26|Rate computation|
|scikit-learn|≥1.3|Per-ruleprecision|



## Input 

joined_evaluation_frame.parquet ← from E1 

Columns used: synthetic_pattern_type, y_true, y_pred, y_score 

generated_alerts.csv ← from P2 

Columns used: triggered_patterns, created_at, transaction_id 

— Output pattern_metrics.json 

json { "per_pattern_detection_rate": { "structuring": 0.88, "circular_laundering": 0.76, "layering_chain": 0.71, "fan_in": 0.82, "fan_out": 0.80, "fraud_ring": 0.69, "dormant_activation": 0.91, "velocity_burst": 0.94, "cross_border_layering": 0.73, "round_tripping": 0.65 }, "per_rule_precision": { "R02_structuring": 0.71, "R10_cycle_closure": 0.89, "R13_benford": 0.67 }, "latency_metrics": { "mean_alert_latency_ms": 142, "p95_alert_latency_ms": 380, "p99_alert_latency_ms": 610 } } 

## — E5 Report Generator 

Purpose: Aggregate all evaluator outputs into a single evaluation_report.json . Compare against external published benchmarks. 

## Libraries 

|Library|Version|Purpose|
|---|---|---|
|json|stdlib|Report serialization|
|pandas|≥2.0|Summary tables|
|matplotlib|≥3.8|PR curve,ROC curve export|
|seaborn|≥0.13|Confusion matrixheatmap|



## Input 

transaction_metrics.json ← from E2 community_metrics.json ← from E3 pattern_metrics.json ← from E4 

## — Output evaluation_report.json 

json { "run_id": "UUID4", "evaluated_at": "2025-01-01T12:00:00Z", "dataset_stats": { "total_stream_transactions": 50000, "total_suspicious_ground_truth": 400, "illicit_ratio": 0.008, "total_alerts_generated": 512 }, "transaction_level_metrics": { "...all fields from E2..." }, "community_level_metrics": { "...all fields from E3..." }, "pattern_coverage": { "...all fields from E4 per_pattern_detection_rate..." }, "rule_performance": { "...all fields from E4 per_rule_precision..." }, "latency_metrics": { "...all fields from E4 latency_metrics..." }, "external_benchmarks": { "tide_li_lightgbm_pr_auc": 78.05, "tide_hi_xgboost_pr_auc": 85.12, "amlworld_multi_gnn_minority_f1": 0.81, "amlworld_mega_gnn_minority_f1": 0.87, "your_minority_f1": 0.77, "your_pr_auc": 83.0, "vs_tide_hi_delta_pr_auc": -2.12, "vs_mega_gnn_delta_minority_f1": -0.10 } } 

## DEPLOYMENTARCHITECTURE 

┌────────────────────────────────────────────────────────────────┐ 

│  System 1 (Generator): runs locally / CI job — one-time       │ 

│  Runtime: Python 3.11+, ~5 min for 500K transactions │ ──────────────────────────────────────────────────────────────── `├ ┤` │  System 2 Offline Pipeline: GPU server (CUDA) for L4A TGN     │ │  Runtime: ~2–4h for 500K transactions, embedding_dim=64 │ ──────────────────────────────────────────────────────────────── `├ ┤` │  System 2 Online API: FastAPI → Render / Railway │ │  Target latency: p99 < 500ms per transaction │ ──────────────────────────────────────────────────────────────── `├ ┤` │  Frontend: Next.js → Vercel                                    │ ──────────────────────────────────────────────────────────────── `├ ┤` │  Alert DB: PostgreSQL → Supabase                               │ ──────────────────────────────────────────────────────────────── `├ ┤` │  Artifact Store: Local disk (dev) / AWS S3 (prod) │ ──────────────────────────────────────────────────────────────── `├ ┤` │  Graph Engine: NetworkX (dev) / Neo4j (prod) │ ──────────────────────────────────────────────────────────────── `├ ┤` │  Account State Cache: in-memory dict (dev) / Redis (prod) │ ──────────────────────────────────────────────────────────────── `├ ┤` │  System 3 (Evaluator): runs locally / CI job post-demo │ └────────────────────────────────────────────────────────────────┘ 

## - - COMPLETE END TO END WORKFLOW 

══ SYSTEM 1 — GENERATOR ══════════════════════════════════════════ 

1.  G1  Build 5,000 synthetic accounts → accounts.parquet 2.  G2  Generate 500K normal transactions → normal_transactions.parquet 3.  G3  Inject 10 laundering scenario types → suspicious_transactions.parquet ground_truth_records.parquet 

4.  G4  Merge + sort + split 90/10 → historical_transactions.csv stream_transactions.csv hidden_ground_truth.csv [SEALED] 

══ SYSTEM 2 — DETECTION PLATFORM — OFFLINE ═══════════════════════ 

5.  S1  Feature engineering → behavioral_features.parquet + feature_scaler.pkl 6.  S2  Build directed multigraph → transaction_multigraph.pkl + indexes 7.  L2A Graph Feature Preprocessor → graph_features.parquet 

8.  L2B Graph Analytics Engine → community_profiles.parquet + suspicious_paths.parquet 

9.  L3A Train Isolation Forest + LOF + Autoencoder → *.pkl + autoencoder.pt 10. L4A Train TGN with link-prediction → tgn_model.pt + node_embeddings.npy 

11. S3  Persist all artifacts → artifacts/ 

══ SYSTEM 2 — DETECTION PLATFORM — ONLINE ════════════════════════ 

12. Load all artifacts from S3 into memory at startup 

13. S4  Emit next TransactionEvent from stream_transactions.csv 

14. S5  Incrementally update behavioral features → LiveFeatureVector 

15. S6  Add edge to multigraph + recompute k-hop features → LiveGraphFeatureVector (also detects cycle closure, updates community risk) 

16. L1  Execute 13 AML rules → RuleEngineOutput (rule_score) 

17. L3B Run IF + LOF + Autoencoder inference → BehavioralAnomalyOutput (anomaly_score) 

18. L4B Update TGN memory + GNN forward pass → GNNInferenceOutput (gnn_score) graph_score ← LiveGraphFeatureVector.sender_community_risk_score 

19. P1  Fuse 4 scores + compute group_risk_score → FusedRiskOutput 

20. P2  If High/Critical: persist Alert → PostgreSQL + generated_alerts.csv 

21. P3  Push live update to dashboard via WebSocket 

22. Repeat from Step 13 for next transaction 

- ══ SYSTEM 3 — EVALUATION ENGINE ══════════════════════════════════ 

23. E1  Join generated_alerts.csv `↔` hidden_ground_truth.csv → joined frame 

- - 

- 24. E2  Compute transaction-level metrics (minority F1, PR AUC, ROC AUC) 

25. E3  Compute community-level metrics (ring detection rate, cluster match) 

26. E4  Compute per-pattern coverage + per-rule precision + latency 

27. E5  Aggregate + compare vs external benchmarks → evaluation_report.json 

