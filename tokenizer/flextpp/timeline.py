# FlexTPP Patient Health Timeline Tokenizer

import pandas as pd
from dataclasses import dataclass
from typing import Any

from .diagnosis import DiagnoseTokenizer
from .procedure import ProcedureTokenizer
from .medication import MedicationTokenizer
from .lab import LabTokenizer
from .blood_pressure import BloodPressureTokenizer
from .demography import DemographyTokenizer


@dataclass
class Event:
    """
    Represents a single event in the patient timeline.

    Attributes:
        time: Relative time in days from session start (e.g., 1.0 = 1 day after admission)
        tokens: Tokenized representation of the event
        event_type: Type of event (LAB, MED, PROC, DIAG, BP, DEMO)
        raw_data: Optional metadata about the event
    """
    time: float
    tokens: str
    event_type: str
    raw_data: dict | None = None


class PatientTimelineTokenizer:
    """
    Combines all tokenizers to create a complete patient health timeline.
    Events are ordered chronologically with continuous time values.
    """

    def __init__(
        self,
        gsn_to_atc: dict | None = None,
        atc_code_to_name: dict | None = None,
    ):
        self.gsn_to_atc = gsn_to_atc or {}
        self.atc_code_to_name = atc_code_to_name or {}

        self.diagnosis_tokenizer = DiagnoseTokenizer()
        self.procedure_tokenizer = ProcedureTokenizer()
        self.medication_tokenizer = MedicationTokenizer(
            code_to_name=self.atc_code_to_name
        )
        self.lab_tokenizer = LabTokenizer()
        self.bp_tokenizer = BloodPressureTokenizer()
        self.demography_tokenizer = DemographyTokenizer()

        self._fitted = False

    def fit(self, tables: dict[str, pd.DataFrame]) -> 'PatientTimelineTokenizer':
        self.diagnosis_tokenizer.build_vocabulary(tables['hosp.diagnoses_icd'])
        self.procedure_tokenizer.build_vocabulary(tables['hosp.procedures_icd'])

        prescriptions = self._prepare_prescriptions(tables['hosp.prescriptions'])
        if len(prescriptions) > 0:
            self.medication_tokenizer.build_vocabulary(
                prescriptions.dropna(subset=['atc_code'])
            )

        self.lab_tokenizer.build_vocabulary(
            tables['hosp.d_labitems'],
            tables['hosp.labevents']
        )
        self.bp_tokenizer.build_vocabulary(tables['hosp.omr'])
        self.demography_tokenizer.build_vocabulary(
            tables['hosp.patients'],
            tables['hosp.admissions']
        )

        self._fitted = True
        return self

    def _prepare_prescriptions(self, prescriptions: pd.DataFrame) -> pd.DataFrame:
        prescriptions = prescriptions.copy()
        prescriptions['gsn'] = prescriptions['gsn'].astype(str).str.zfill(6)
        prescriptions = prescriptions[prescriptions['gsn'] != '000nan']

        if self.gsn_to_atc:
            prescriptions['atc_code'] = prescriptions['gsn'].map(self.gsn_to_atc)
        else:
            prescriptions['atc_code'] = None

        return prescriptions

    def _to_relative_time(self, timestamp: pd.Timestamp, session_start: pd.Timestamp) -> float:
        """Convert absolute timestamp to relative time in days from session start."""
        delta = timestamp - session_start
        return delta.total_seconds() / (24 * 60 * 60)  # Convert to days

    def get_session_events(
        self,
        hadm_id: int,
        tables: dict[str, pd.DataFrame],
    ) -> list[Event]:
        admissions = tables['hosp.admissions']
        admission = admissions[admissions['hadm_id'] == hadm_id].iloc[0]

        admit_time = pd.to_datetime(admission['admittime'])
        discharge_time = pd.to_datetime(admission['dischtime'])

        events = []

        # Lab events
        labevents = tables['hosp.labevents']
        session_labs = labevents[
            (labevents['hadm_id'] == hadm_id) &
            (labevents['charttime'].notna())
        ].copy()
        session_labs['charttime'] = pd.to_datetime(session_labs['charttime'])
        session_labs = session_labs[
            (session_labs['charttime'] >= admit_time) &
            (session_labs['charttime'] <= discharge_time)
        ]

        tokenized_labs = self.lab_tokenizer.tokenize(session_labs)
        for _, row in tokenized_labs.iterrows():
            if pd.notna(row.get('tokenized_version')):
                events.append(Event(
                    time=self._to_relative_time(row['charttime'], admit_time),
                    tokens=row['tokenized_version'],
                    event_type='LAB',
                    raw_data={'itemid': row['itemid'], 'valuenum': row.get('valuenum')}
                ))

        # Prescriptions
        prescriptions = tables['hosp.prescriptions']
        session_meds = prescriptions[
            (prescriptions['hadm_id'] == hadm_id) &
            (prescriptions['starttime'].notna())
        ].copy()
        session_meds['starttime'] = pd.to_datetime(session_meds['starttime'])
        session_meds = session_meds[
            (session_meds['starttime'] >= admit_time) &
            (session_meds['starttime'] <= discharge_time)
        ]

        session_meds = self._prepare_prescriptions(session_meds)
        session_meds_with_atc = session_meds.dropna(subset=['atc_code'])

        if len(session_meds_with_atc) > 0:
            tokenized_meds = self.medication_tokenizer._divide_code_into_parts(
                session_meds_with_atc
            )
            for _, row in tokenized_meds.iterrows():
                if pd.notna(row.get('tokenized_version')) and row['tokenized_version']:
                    events.append(Event(
                        time=self._to_relative_time(row['starttime'], admit_time),
                        tokens=row['tokenized_version'],
                        event_type='MED',
                        raw_data={'drug': row.get('drug'), 'atc_code': row['atc_code']}
                    ))

        # Procedures
        procedures = tables['hosp.procedures_icd']
        session_procs = procedures[
            (procedures['hadm_id'] == hadm_id) &
            (procedures['chartdate'].notna())
        ].copy()
        session_procs['chartdate'] = pd.to_datetime(session_procs['chartdate'])
        session_procs = session_procs[
            (session_procs['chartdate'] >= admit_time.normalize()) &
            (session_procs['chartdate'] <= discharge_time.normalize())
        ]

        tokenized_procs = self.procedure_tokenizer._divide_code_into_parts(session_procs)
        for _, row in tokenized_procs.iterrows():
            if pd.notna(row.get('tokenized_version')):
                events.append(Event(
                    time=self._to_relative_time(row['chartdate'], admit_time),
                    tokens=row['tokenized_version'],
                    event_type='PROC',
                    raw_data={'icd_code': row['icd_code'], 'icd_version': row.get('icd_version')}
                ))

        # Diagnoses (at admission time = 0.0)
        diagnoses = tables['hosp.diagnoses_icd']
        session_diags = diagnoses[diagnoses['hadm_id'] == hadm_id].copy()
        session_diags = session_diags.sort_values('seq_num')

        tokenized_diags = self.diagnosis_tokenizer._divide_code_into_parts(session_diags)
        for _, row in tokenized_diags.iterrows():
            if pd.notna(row.get('tokenized_version')):
                events.append(Event(
                    time=0.0,  # Diagnoses at admission
                    tokens=row['tokenized_version'],
                    event_type='DIAG',
                    raw_data={'icd_code': row['icd_code'], 'seq_num': row['seq_num']}
                ))

        # Sort by time
        events.sort(key=lambda e: e.time)
        return events

    def tokenize_session(
        self,
        hadm_id: int,
        tables: dict[str, pd.DataFrame],
        include_demography: bool = True,
    ) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fitted before tokenizing. Call fit() first.")

        admissions = tables['hosp.admissions']
        patients = tables['hosp.patients']

        admission = admissions[admissions['hadm_id'] == hadm_id].iloc[0]
        subject_id = admission['subject_id']
        admit_time = pd.to_datetime(admission['admittime'])
        discharge_time = pd.to_datetime(admission['dischtime'])

        result = {
            'hadm_id': hadm_id,
            'subject_id': subject_id,
            'admit_time': admit_time,
            'discharge_time': discharge_time,
            'events': [],
        }

        # Demography event at time 0.0
        if include_demography:
            demo_df = self.demography_tokenizer.tokenize(
                patients[patients['subject_id'] == subject_id],
                admissions[admissions['hadm_id'] == hadm_id]
            )
            if len(demo_df) > 0 and pd.notna(demo_df.iloc[0]['tokenized_version']):
                result['events'].append(Event(
                    time=0.0,
                    tokens=demo_df.iloc[0]['tokenized_version'],
                    event_type='DEMO',
                ))

        # Get all session events
        events = self.get_session_events(hadm_id, tables)
        result['events'].extend(events)

        # Sort all events by time
        result['events'].sort(key=lambda e: e.time)

        return result

    def get_combined_vocabulary(self) -> dict[str, int]:
        combined = {}
        idx = 0

        for vocab in [
            self.diagnosis_tokenizer.vocabulary,
            self.procedure_tokenizer.vocabulary,
            self.medication_tokenizer.vocabulary,
            self.lab_tokenizer.vocabulary,
            self.bp_tokenizer.vocabulary,
            self.demography_tokenizer.vocabulary,
        ]:
            for token in vocab:
                if token not in combined:
                    combined[token] = idx
                    idx += 1

        return combined
