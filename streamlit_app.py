"""
streamlit_app.py - the FailScope demo.
Run from the FailScope folder:  streamlit run streamlit_app.py
"""
import matplotlib.pyplot as plt
import streamlit as st

from backend.app import demo
from backend.app.agent.agent import MIN_CONFIDENCE
from backend.app.sim.simulator import Simulator

st.set_page_config(page_title="FailScope", page_icon="🔎", layout="wide")


@st.cache_resource
def get_sim():
    return Simulator()


sim = get_sim()
LAYER_COLOR = {"data": "blue", "model": "violet", "system": "orange"}


# ---------- button callbacks (they run before the page redraws) ----------
def start():
    fault, variant = demo.SCENARIOS[st.session_state["scenario"]]
    reasoner, note = demo.make_reasoner(st.session_state["use_llm"])
    st.session_state["inc"] = demo.start_incident(sim, fault, variant,
                                                  int(st.session_state["seed"]), reasoner)
    st.session_state["note"] = note


def approve():
    demo.apply_fix(sim, st.session_state["inc"])


def decline():
    demo.decline(st.session_state["inc"])


# ---------- sidebar ----------
with st.sidebar:
    st.header("Inject a fault")
    st.selectbox("Scenario", list(demo.SCENARIOS), key="scenario")
    st.number_input("Random seed", min_value=0, max_value=10000, value=7, step=1, key="seed")
    st.checkbox("Use the LLM reasoner (needs an API key)", value=False, key="use_llm")
    st.button("Inject fault and investigate", type="primary", on_click=start)
    st.caption("The agent never sees which fault was injected. It only sees telemetry, "
               "through six deterministic tools.")

# ---------- main page ----------
st.title("FailScope")
st.caption("An AI agent that investigates ML system failures, proposes a fix, "
           "and verifies that the system really recovered.")

inc = st.session_state.get("inc")
if inc is None:
    st.info("Pick a scenario in the sidebar and press **Inject fault and investigate**.")
    st.stop()

state = demo.status(inc)
report = inc.report
st.caption(st.session_state.get("note", ""))

# 1. telemetry
st.subheader("1. Telemetry")
fig = demo.make_chart(inc)
st.pyplot(fig)
plt.close(fig)

if state == "healthy":
    st.success("No incident detected. All signals are within their normal range.")
    st.stop()

# 2. investigation
st.subheader("2. Investigation")
st.caption(f"Reasoner used: {inc.reasoner_source}")
with st.expander("What the agent did, step by step"):
    for line in report.trace:
        st.write("- " + line)
by_layer = demo.evidence_by_layer(inc)
cols = st.columns(3)
for col, layer in zip(cols, ["data", "model", "system"]):
    with col:
        st.markdown(f":{LAYER_COLOR[layer]}[**{layer.upper()} layer**]")
        for e in by_layer[layer]:
            icon = "⚠️" if e.anomalous else "✅"
            st.markdown(f"{icon} `{e.id}` {e.finding}")

# 3. diagnosis
st.subheader("3. Diagnosis")
top = report.top
if top is None:
    st.error("No hypothesis passed the validator. This needs a human.")
else:
    st.success(f"Most likely root cause: **{top.cause}** (confidence {top.confidence:.2f})")
    st.write(top.reasoning)
    st.caption("Cited evidence: " + ", ".join(top.evidence_ids))
if len(report.valid) > 1:
    with st.expander("Other candidates that passed the validator"):
        for h in report.valid[1:]:
            st.write(f"**{h.cause}** (confidence {h.confidence:.2f}): {h.reasoning}")
if report.rejected:
    with st.expander("Rejected hypotheses"):
        for r in report.rejected:
            why = "; ".join(r.reasons) if r.reasons else f"confidence below {MIN_CONFIDENCE}"
            st.write(f"**{r.hypothesis.cause}** (confidence {r.hypothesis.confidence:.2f}): {why}")

# 4. fix and verify
st.subheader("4. Fix and verify")
for i, a in enumerate(inc.attempts, start=1):
    message = f"Attempt {i}: `{a.hypothesis.action}` (for {a.hypothesis.cause}). {a.verification.summary}"
    if a.verification.recovered:
        st.success(message)
    else:
        st.error(message)
        st.caption("This fix was undone before trying the next suggestion.")
    with st.expander(f"Verification numbers for attempt {i}"):
        st.json(a.verification.details)

if state == "awaiting_approval":
    h = demo.pending(inc)
    st.info(f"Proposed action: **{h.action}** (because: {h.cause}). "
            "Nothing changes until you approve.")
    c1, c2 = st.columns(2)
    c1.button("Approve fix", type="primary", on_click=approve)
    c2.button("Decline", on_click=decline)
elif state == "recovered":
    st.success("System recovered. Verified against the healthy baseline.")
elif state == "declined":
    st.warning("Fix declined. The incident stays open for a human.")
else:
    st.error("Could not recover automatically. Escalated to a human.")