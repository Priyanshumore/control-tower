# Integrated Manufacturing Operations Control Tower

A multi agent planning dashboard that runs the full chain from demand history to a shop
floor Gantt chart: forecasting, inventory planning, MPS, BOM explosion, MRP, order release,
finite capacity scheduling, KPI comparison, exception management and scenario simulation.

A coordinator agent takes the request, delegates to ten specialists, and assembles the
answer. The specialists also talk to each other while the plan is being built, so the
transcript in the app is the record of how the plan was actually decided.

The worked instance is themed on Nestle Maggi. The engine itself is dataset agnostic: point
it at any planning dataset and it maps the columns, runs the same thirteen modules and
redraws every chart.

---

## 1. Running it locally

Requires Python 3.9 or newer.

```bash
cd moct
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Nothing leaves the machine: there is no API
call, no database and no network dependency at runtime.

Shortcuts: `./run.sh` on macOS or Linux, `run.bat` on Windows. Both create the virtual
environment, install the requirements and start the app.

To rebuild the bundled dataset from scratch:

```bash
python scripts/generate_dataset.py
```

---

## 2. Putting it on GitHub and Streamlit Cloud

The repository is already initialised with a first commit, so publishing it is three
commands. Create an empty repository on GitHub first (no README, no licence, no gitignore,
since this project already has them), then:

```bash
cd maggi-control-tower
git remote add origin https://github.com/<your-username>/maggi-control-tower.git
git branch -M main
git push -u origin main
```

If GitHub asks for a password, use a personal access token rather than your account
password: GitHub Settings, Developer settings, Personal access tokens, fine grained token
with Contents read and write on this repository.

Then deploy:

1. Go to `share.streamlit.io` and sign in with GitHub.
2. Choose **Create app**, then **Deploy a public app from GitHub**.
3. Repository `<your-username>/maggi-control-tower`, branch `main`, main file path `app.py`.
4. Under Advanced settings choose Python 3.12.
5. Deploy. The first build takes two or three minutes while the dependencies install.

The app needs no secrets, no database and no API key, so there is nothing to configure
after the deploy. `requirements.txt` pins Streamlit 1.63 or newer, which matters because
charts and tables only default to full container width from that version.

Everything the app writes at runtime goes to `exports/`, which is disposable. Streamlit
Cloud restarts containers freely, so treat exports as downloads rather than storage: use
the Reports tab to pull the workbook down.

A note on the free tier: the app sleeps after a period without traffic and takes a few
seconds to wake. That is normal and not a fault in the build.

---

## 3. What each tab does

The tabs are numbered to match the thirteen steps of the assignment brief.

| Tab | Step | What it produces |
|---|---|---|
| Control tower | overview | Plan verdict, demand against supply, work centre load heat map, ranked actions |
| Agents | architecture | The roster, a prompt box, and the transcript of how each answer was assembled |
| 1-2 Data | 1, 2 | Table detection, column mapping, validation issues, raw against mapped preview |
| 3 Forecast | 3 | Six methods fitted per product, hold out accuracy, method selection, forecast chart |
| 4 Inventory | 4 | Safety stock, reorder point, EOQ, ABC, weeks of cover, projected stockouts |
| 5 MPS | 5 | Master schedule grid, PAB, ATP, lot rules, capacity levelling, feasibility |
| 6 BOM | 6 | Indented bill, extended quantities with scrap, gross requirements by level |
| 7 MRP | 7 | Time phased record per item, netting, lot sizing, lead time offset, pegging |
| 8 Release | 8 | Released against held works orders and the component that blocks each one |
| 9 Capacity | 9 | Available hours per work centre, routing, executable job list with lot splitting |
| 10 Schedule | 10 | Finite capacity simulation, Gantt chart, utilisation, completion against due date |
| 11 KPIs | 11 | All seven dispatching rules compared on thirteen measures, weighted ranking |
| 12 Exceptions | 12 | Six agents plus a supervisor, ranked by severity and financial exposure, decisions |
| 13 Scenarios | 13 | Five disruptions run against the baseline, side by side |
| Reports | 13.4 | One click Excel workbook of every table, or any single table as CSV |

Everything is one pass of one pipeline. Changing a parameter in the control panel on the
left flows all the way through to the Gantt chart.

---

## 4. The agent architecture

```
                                  User prompt
                                       |
                                 COORDINATOR
              reads the request, delegates, assembles the answer
                                       |
   +--------+--------+--------+--------+--------+--------+--------+--------+
   |        |        |        |        |        |        |        |        |
 Data   Forecast Inventory Capacity   MPS   Supplier Material Scheduling Exceptions
                                                                            |
                                                                        Scenario
```

Ten sub agents, one per part of the brief. Each owns its module, answers questions on its
own topics, and may consult another agent before answering.

| Agent | Owns | Answers on |
|---|---|---|
| Data | Steps 1 and 2 | dataset, mapping, data quality |
| Forecast | Step 3 | method selection, accuracy, the demand signal |
| Inventory | Step 4 | safety stock, reorder point, EOQ, cover, risk |
| Capacity | Steps 9 and 5.5 | available hours, the constraint, load by work centre |
| MPS | Step 5 | master schedule, build ahead, feasibility |
| Supplier | Step 7 support | lead times, reliability, expedite quotes |
| Material | Steps 6 and 7 | BOM, netting, shortages, pegging |
| Scheduling | Steps 8, 10 and 11 | release, sequencing, lateness, rule choice |
| Exceptions | Step 12 | ranking what is wrong, obtaining the remedy |
| Scenario | Step 13 | disruptions against the live plan |

### The agents genuinely interact

Three of these are not commentators bolted onto a finished plan. They are consulted at the
decision points inside the run, through the same message bus:

- **The master schedule cannot help itself to capacity.** During levelling it has to ask the
  capacity agent how many hours are free on a work centre in a period, and the capacity
  agent grants or refuses. On the bundled dataset that is 24 grants totalling about 100
  hours in a normal run.
- **Order release asks the material agent** whether each component exists in the release
  period rather than reading the MRP table itself, and the material agent reports back which
  ones are short.
- **The material agent consults the supplier agent** before it will promise an expedite:
  whether the supplier is reliable enough, what the pulled in lead time would be, and what
  the minimum order does to the quantity.

Question handling chains in the same way. Ask why something is late and the scheduling agent
asks the capacity agent for the constraint and the material agent for shortages before
answering. Ask what to do about a shortage and the exception agent takes it to the agent that
owns that part of the plan, which may in turn go to the supplier. Ask about one product and
the coordinator fans out to every specialist and assembles a briefing.

The consequence worth stating: **the agent driven run and a direct pipeline run produce
identical numbers.** The agents change who decides and how it is recorded, not the
arithmetic.

### Prompting the agents

The Agents tab takes plain language. The parser is deterministic and domain specific rather
than a language model, so it runs offline, costs nothing and gives the same answer twice.
It recognises item codes, work centres, periods, percentages, hours, dispatching rules,
severities and scenario phrasing.

```
How is the plan looking?                     -> every specialist reports, coordinator assembles
Why is anything running late?                -> Scheduling, consulting Capacity and Material
Where is the bottleneck?                     -> Capacity
Tell me everything about FG-MAG-70           -> a briefing from all ten
What do I do about RM-CHILLI?                -> Exceptions -> Material -> Supplier
What if WC-SHEET goes down for 60 hours?     -> Scenario, then a comparison table
Who supplies RM-PALMOIL?                     -> Supplier
```

Driving it from a script instead of the browser:

```python
from engine.loader import load_folder
from engine.agentic import PlanningSession

session = PlanningSession(load_folder("data/nestle_maggi"))
session.run()

reply = session.ask("why is anything running late?")
print(reply.headline)
for msg in session.bus.transcript()[reply.from_seq:]:
    print(msg.arrow(), msg.topic, msg.text)
```

---

## 5. The two bundled plants

Any folder placed in `data/` becomes a selectable dataset, so both of these appear in the
control panel and a third plant can be added without touching a line of code.

| | Nestle Maggi | Vayu Mobility |
|---|---|---|
| Process | food, batch and continuous | discrete assembly |
| Products | 4 noodle and sauce SKUs | 3 vehicles plus a service kit |
| Items | 28 across 4 BOM levels | 36 across 4 BOM levels |
| Periods | 1 to 48 | fiscal weeks 201 to 248 |
| Vocabulary | FMCG: Item, Demand, Work_Centre | engineering: Part_Code, Despatched_Units, Cost_Centre |
| Constraint | sheeting line, WC-SHEET | end of line testing, WC-TEST |
| Character | fits after levelling, everything on time | fits after levelling, three jobs still run late |

Vayu Mobility is a fictitious company. It exists to prove the control tower is not quietly
tuned to the first dataset: nothing in its file names, sheet names, column headers or period
numbering matches the Maggi data, and it still maps end to end with no manual correction.

`exports/Vayu_Mobility_Plant_Data.xlsx` is the same data as one Excel workbook, eleven
sheets, for testing the upload path. Attach it under "Attach my own files" and the whole
dashboard rebuilds around it. The workbook, the CSV folder, a zip of the CSVs and the loose
CSVs all produce an identical plan.

To rebuild either one:

```bash
python scripts/generate_dataset.py         # Nestle Maggi
python scripts/generate_vayu_dataset.py    # Vayu Mobility
python scripts/build_vayu_workbook.py      # the Excel version
```

---

## 6. Attaching your own dataset

Choose "Attach my own files" in the control panel and upload CSVs, Excel workbooks (each
sheet is read as a table), JSON, or a single zip containing any of those.

**How the matching works.** Each file is scored against all eleven canonical tables using
filename hints plus how much of that table's column shape the file actually carries. Tables
are then handed out best pair first, so an early file cannot claim a table that a later file
fits better. Within a table, every canonical field is scored against every column and the
strongest pairs are assigned first, which is what stops `ORDER_COST` being taken by
`unit_cost` while `ordering_cost`, which matches it exactly, goes unmapped.

Headers are matched on exact names, synonyms, word tokens and expanded abbreviations, so
`WORK_CTR` finds `work_centre`, `COMP_MATL` finds `component` and `PLND_DELIV_TIME` finds
`lead_time` without any configuration.

**Anything it gets wrong is fixable.** The Data tab exposes every field as a dropdown over
the real column names in your file. Correct it, press "Apply mapping and replan", and the
whole dashboard recomputes.

### Tables it looks for

Only the first two are required. Missing optional tables degrade gracefully: without a BOM
the plan runs at finished goods level, without routing and capacity the scheduling tabs
explain what is missing instead of failing.

| Table | Required | Key fields |
|---|---|---|
| Demand history | yes | item, period, demand |
| Inventory master | yes | item, on hand, safety stock, lead time, lot rule, costs |
| Customer orders | no | order id, item, period, qty, priority |
| Bill of material | no | parent, component, qty per, scrap % |
| Supplier lead time | no | supplier, item, lead time, reliability, MOQ |
| Routing and work centre | no | item, op sequence, work centre, setup, run time |
| Machine capacity | no | work centre, machines, hours, shifts, days, efficiency |
| Production orders | no | order id, item, qty, release and due period |
| Scheduled receipts | no | item, period, qty |
| MPS template | no | item, period, mps qty |
| Period calendar | no | period, start date |

Period numbering does not have to start anywhere in particular. A dataset numbered 1001 to
1048 plans identically to the same dataset numbered 1 to 48.

---

## 7. The formulas

**Forecasting.** Naive, moving average, weighted moving average, single exponential
smoothing, Holt's linear trend and Holt Winters additive seasonality, all written from
first principles. Parameters are grid searched on the training block, every method is then
ranked on a hold out block it has never seen, and the winner is refitted on the full history
before projecting. Accuracy is reported as MAD, MSE, RMSE, MAPE, bias and tracking signal.

**Inventory.**

```
safety stock  = z x sigma(demand) x sqrt(lead time)
reorder point = average demand x lead time + safety stock
EOQ           = sqrt(2 x annual demand x ordering cost / (unit cost x holding rate))
ABC           = Pareto on annual usage value, 80% and 95% cut offs
```

**MPS.** Gross requirement follows the demand time fence: inside the fence, confirmed
customer orders only; outside it, the configured rule (default is the greater of forecast
and orders). Then

```
PAB = opening balance + master schedule + scheduled receipts - gross requirement
ATP = supply in this lot - orders committed before the next lot
```

Lot rules supported: lot for lot, fixed order quantity, EOQ and periodic order quantity.
Capacity levelling only ever pulls volume earlier, never later, so customer due dates are
protected, and each move is checked against every work centre on the item's routing.

**MRP.** Items are processed in low level code order so a component shared by several
parents is netted once, after all its parents are planned.

```
net requirement      = gross requirement - scheduled receipts - opening available + safety stock
planned order receipt = net requirement rounded up by the lot rule
planned order release = planned receipt shifted back by the lead time
```

**Capacity and scheduling.**

```
available hours = machines x hours per shift x shifts per day x days per week x efficiency
processing time = (setup + quantity x run time per unit) / (efficiency x availability)
```

The scheduler is a non delay simulation: at every decision point it picks from the operations
whose predecessor is complete and whose machine is free, using the chosen dispatching rule.
Rules available: FCFS, SPT, LPT, EDD, critical ratio, minimum slack and customer priority.
Large orders are split into sub lots so operations can overlap.

**KPIs.** Makespan, flow time, waiting time, lateness, tardiness, jobs tardy, on time
delivery, utilisation, throughput, average WIP, takt time and setup share. The weighted
ranking leans on tardiness and on time delivery first, then flow time, makespan and
utilisation.

---

## 8. Code layout

```
moct/
  app.py                  Streamlit dashboard, fourteen tabs
  engine/
    schema.py             canonical tables and fields, table and column matching
    loader.py             reads csv, excel, json, zip; applies mappings
    forecasting.py        six methods and the accuracy metrics
    inventory.py          safety stock, ROP, EOQ, ABC, projection
    bom.py                low level codes, indented bill, explosion
    mps.py                master schedule, lot rules, ATP, capacity levelling
    mrp.py                netting, lot sizing, offsetting, pegging
    capacity.py           available hours, rough cut capacity load
    scheduling.py         job build, lot splitting, non delay simulation, seven rules
    kpis.py               schedule KPIs, rule comparison, Gantt shaping
    agents.py             six specialist agents, supervisor ranking, decision overrides
    pipeline.py           runs all modules in order, scenario library, run comparison
    ui.py                 theme, plotly template, shared render helpers
    agentic/
      bus.py              message bus and shared blackboard
      base.py             the contract every agent keeps
      roster.py           the ten specialists
      coordinator.py      coordinator, scenario agent, planning session
      language.py         prompt parsing into intents and entities
  scripts/
    generate_dataset.py   builds the bundled synthetic dataset
  data/nestle_maggi/      eleven input CSVs
  exports/                workbook output
```

The engine has no Streamlit imports, so the same pipeline can be driven from a script or a
notebook, either through the agents as shown above or directly:

```python
from engine.loader import load_folder
from engine.pipeline import run_pipeline, Settings

ds  = load_folder("data/nestle_maggi")
res = run_pipeline(ds, Settings(horizon=12, dispatch_rule="EDD"))
print(res["kpi_summary"])
```

---

## 9. How it was checked

Fifty two audits were run while building this, each one hand verified rather than assumed.
Forecast arithmetic was recomputed by hand for every method; Holt Winters recovers a clean
seasonal series at 0.83% MAPE. MRP row arithmetic was checked across all 28 items with lead
time offsets and pegging matched to manual explosion. Every schedule was validated for
machine overlap, precedence and arrival time under all seven rules.

Adding the second plant paid for itself immediately: it exposed a latent bug where the
scheduling agent read a column called `tardiness` when the real name is `tardiness_min`. The
Maggi plan delivers everything on time, so that line had never once been reached. It also
showed the column matcher had been quietly tuned to FMCG and SAP vocabulary, and that a
three letter file hint was matching the letters in the middle of `Works_Orders` and reading
shop orders as sales orders.

Eight real defects were found and fixed along the way: a depth first BOM traversal that lost
shared component contributions, a pandas text casting issue, a capacity reference that double
counted machines, capacity levelling that ignored work centres outside the one it was
relieving, a crash on datasets with no capacity table, a crash when accepting an
exception that carries no period, a scheduling agent reading a column name that does not
exist, and a file hint short enough to match inside an unrelated word.

The multi agent layer was checked the same way. The agent driven run reproduces the direct
pipeline run exactly, field for field, so the roster changes who decides and how it is
recorded without touching the arithmetic. Every interaction claimed in section 4 was then
asserted against the live message log: that the master schedule really does ask the capacity
agent before building ahead, that release really does ask the material agent about
components, that the material agent really does consult the supplier agent before quoting an
expedite, and that a question about one product really does fan out to seven specialists.
All eleven pass, and the counters agree with the plan: the two components the material agent
reported short are the two orders the release step held. Prompt routing was checked across
thirty two phrasings, including the awkward ones where a topic word appears inside a what if
question.

The strongest check is invariance. The bundled dataset was rewritten with ERP style file
names, unrecognisable column headers, injected junk columns and every period shifted by
1000. The control tower produced a bit identical plan: all sixteen KPIs, and the MPS, MRP,
release, job list, schedule, exception and rule comparison tables all matched once the period
offset and identifier labels were normalised.

---

## 10. Data note

The bundled Nestle Maggi dataset is **synthetic**. Product names and the process flow are
modelled on publicly understood instant noodle and ketchup manufacturing. Every quantity,
cost, lead time, capacity and demand figure is generated by a seeded random model in
`scripts/generate_dataset.py` for teaching purposes. None of it is Nestle company data and
none of it should be read as a statement about Nestle's actual operations.


## Interactive planning workspace

The sidebar now exposes manual alpha, beta, gamma, moving-average window and
weighted-average weights. Choose a method explicitly to explore its parameters;
Auto compares methods. Disable manual parameters to search candidates. Forecast
changes feed inventory, MPS, MRP, release, scheduling, exceptions and reports.
Flat future values are mathematically expected for level-only methods.

In **Data → Edit planning data**, edit any recognized source table and select
**Apply data and replan**. Validation runs before replacing the current dataset.
Edits are session-only; download the source CSV to retain them. The MPS template
is used only when **Use MPS template quantities** is enabled; entered zeros are
respected, missing item/period pairs are calculated, and capacity levelling is
disabled to preserve manual quantities. Item-master lot rules still govern MRP;
the sidebar lot-policy override governs the master schedule.

Click any card under **Agents** to inspect its role, capabilities, current output
and the latest 30 relevant message exchanges. These are deterministic planning
agents, as in the original project.

Local preview in the prepared environment:

```bash
.venv-runtime/bin/streamlit run app.py
.venv-runtime/bin/python -m unittest discover -s tests -v
```

The light workspace theme uses teal for controls, amber for risk and red for
critical exceptions. Changing inputs invalidates cached plans and clears stale
scenario comparisons.

### Focused agent answers and appearance

Questions now open an **Agent answer** window automatically. The latest answer
also stays above the roster, with a button to reopen it; older answers and the
message trace are collapsed. Stockout queries return one highlighted card per
affected product, including its first shortage period, quantity and unit. These
are explicitly labelled as projections without new production. Complete result
tables are shown rather than silently truncating answers to fourteen rows.

The workspace uses a single light palette so native controls, custom cards, and
Plotly charts remain readable together. Streamlit's appearance menu may still
exist in the host shell, but it is intentionally not used by the workspace.
