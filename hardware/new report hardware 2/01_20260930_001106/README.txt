PHASE 6C REMOTE CAPTURE PACKAGE

FILES
- ppg_samples.csv: raw PPG samples.
- bp_events.csv: timestamped reference-BP/activity events.
- session_metadata.json: session metadata and synchronization definition.
- hardware_audit.json / hardware_audit.txt: acquisition audit.

SYNCHRONIZATION
The primary shared clock is host_timestamp_ms, generated on the capture laptop for every serial sample and every operator event. host_monotonic_ns is also stored for local timing diagnostics.

REFERENCE-BP PROCEDURE
1. Press START REFERENCE BP.
2. Perform the reference cuff measurement according to the device instructions.
3. Press RECORD BP RESULT and enter SBP and DBP.

Do not edit the CSV files before sending them. Do not resave them in Excel.

This package is for research/engineering use and does not establish clinical BP accuracy.
