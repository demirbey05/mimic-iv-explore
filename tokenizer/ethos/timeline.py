# ETHOS Patient Health Timeline Tokenizer
# Combines all tokenizers to generate a complete chronological patient timeline

import pandas as pd
from dataclasses import dataclass
from typing import Any

from .diagnosis import DiagnoseTokenizer
from .procedure import ProcedureTokenizer
from .medication import MedicationTokenizer
from .lab import LabTokenizer
from .blood_pressure import BloodPressureTokenizer
from .demography import DemographyTokenizer
from .time_interval import TimeIntervalTokenizer


@dataclass
class TimelineEvent:
    """Represents a single event in the patient timeline."""
    timestamp: pd.Timestamp
    event_type: str
    tokens: str
    raw_data: dict


class PatientTimelineTokenizer:
    """
    Combines all tokenizers to create a complete patient health timeline.
    Events are ordered chronologically with time interval tokens inserted
    when gaps exceed 5 minutes.
    """

    def __init__(
        self,
        n_quantiles: int = 10,
        gsn_to_atc: dict | None = None,
        atc_code_to_name: dict | None = None,
    ):
        self.n_quantiles = n_quantiles
        self.gsn_to_atc = gsn_to_atc or {}
        self.atc_code_to_name = atc_code_to_name or {}

        # Initialize individual tokenizers
        self.diagnosis_tokenizer = DiagnoseTokenizer()
        self.procedure_tokenizer = ProcedureTokenizer()
        self.medication_tokenizer = MedicationTokenizer(
            code_to_name=self.atc_code_to_name
        )
        self.lab_tokenizer = LabTokenizer(n_quantiles=n_quantiles)
        self.bp_tokenizer = BloodPressureTokenizer(n_quantiles=n_quantiles)
        self.demography_tokenizer = DemographyTokenizer()
        self.time_interval_tokenizer = TimeIntervalTokenizer()

        self._fitted = False

    def fit(self, tables: dict[str, pd.DataFrame]) -> 'PatientTimelineTokenizer':
        """
        Fit all tokenizers on the provided tables.
        tables should be the dict returned by load_mimic_tables().
        """
        # Build vocabularies for all tokenizers
        self.diagnosis_tokenizer.build_vocabulary(tables['hosp.diagnoses_icd'])
        self.procedure_tokenizer.build_vocabulary(tables['hosp.procedures_icd'])

        # Prepare prescriptions with ATC codes
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
        self.time_interval_tokenizer.build_vocabulary()

        self._fitted = True
        return self

    def _prepare_prescriptions(self, prescriptions: pd.DataFrame) -> pd.DataFrame:
        """Prepare prescriptions dataframe with ATC codes."""
        prescriptions = prescriptions.copy()
        prescriptions['gsn'] = prescriptions['gsn'].astype(str).str.zfill(6)
        prescriptions = prescriptions[prescriptions['gsn'] != '000nan']

        if self.gsn_to_atc:
            prescriptions['atc_code'] = prescriptions['gsn'].map(self.gsn_to_atc)
        else:
            prescriptions['atc_code'] = None

        return prescriptions

    def get_session_events(
        self,
        hadm_id: int,
        tables: dict[str, pd.DataFrame],
    ) -> list[TimelineEvent]:
        """
        Get all events for a hospital session (admission) ordered chronologically.
        """
        admissions = tables['hosp.admissions']
        admission = admissions[admissions['hadm_id'] == hadm_id].iloc[0]

        admit_time = pd.to_datetime(admission['admittime'])
        discharge_time = pd.to_datetime(admission['dischtime'])
        subject_id = admission['subject_id']

        events = []

        # 1. Lab events
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
                events.append(TimelineEvent(
                    timestamp=row['charttime'],
                    event_type='LAB',
                    tokens=row['tokenized_version'],
                    raw_data={'itemid': row['itemid'], 'valuenum': row.get('valuenum')}
                ))

        # 2. Prescriptions (medications)
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

        # Add ATC codes
        session_meds = self._prepare_prescriptions(session_meds)
        session_meds_with_atc = session_meds.dropna(subset=['atc_code'])

        if len(session_meds_with_atc) > 0:
            tokenized_meds = self.medication_tokenizer._divide_code_into_parts(
                session_meds_with_atc
            )
            for _, row in tokenized_meds.iterrows():
                if pd.notna(row.get('tokenized_version')) and row['tokenized_version']:
                    events.append(TimelineEvent(
                        timestamp=row['starttime'],
                        event_type='MED',
                        tokens=row['tokenized_version'],
                        raw_data={'drug': row.get('drug'), 'atc_code': row['atc_code']}
                    ))

        # 3. Procedures (use chartdate, assign to start of day)
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
                events.append(TimelineEvent(
                    timestamp=row['chartdate'],
                    event_type='PROC',
                    tokens=row['tokenized_version'],
                    raw_data={'icd_code': row['icd_code'], 'icd_version': row.get('icd_version')}
                ))

        # 4. Diagnoses (no timestamp, place at admission time, ordered by seq_num)
        diagnoses = tables['hosp.diagnoses_icd']
        session_diags = diagnoses[diagnoses['hadm_id'] == hadm_id].copy()
        session_diags = session_diags.sort_values('seq_num')

        tokenized_diags = self.diagnosis_tokenizer._divide_code_into_parts(session_diags)
        for _, row in tokenized_diags.iterrows():
            if pd.notna(row.get('tokenized_version')):
                events.append(TimelineEvent(
                    timestamp=admit_time,  # Place at admission
                    event_type='DIAG',
                    tokens=row['tokenized_version'],
                    raw_data={'icd_code': row['icd_code'], 'seq_num': row['seq_num']}
                ))

        # Sort all events by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events

    def tokenize_session(
        self,
        hadm_id: int,
        tables: dict[str, pd.DataFrame],
        include_demography: bool = True,
        include_time_intervals: bool = True,
    ) -> dict[str, Any]:
        """
        Tokenize a complete hospital session.

        Returns:
            dict with:
                - 'hadm_id': hospital admission ID
                - 'subject_id': patient ID
                - 'admit_time': admission timestamp
                - 'discharge_time': discharge timestamp
                - 'demography_tokens': demographic tokens (if include_demography)
                - 'events': list of TimelineEvent objects
                - 'timeline_tokens': list of all tokens in chronological order
                - 'full_sequence': complete token sequence as a single string
        """
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
            'timeline_tokens': [],
            'full_sequence': '',
        }

        # Demography tokens (at the start)
        if include_demography:
            demo_df = self.demography_tokenizer.tokenize(
                patients[patients['subject_id'] == subject_id],
                admissions[admissions['hadm_id'] == hadm_id]
            )
            if len(demo_df) > 0 and pd.notna(demo_df.iloc[0]['tokenized_version']):
                result['demography_tokens'] = demo_df.iloc[0]['tokenized_version']
                result['timeline_tokens'].append(result['demography_tokens'])

        # Get all events
        events = self.get_session_events(hadm_id, tables)
        result['events'] = events

        # Build timeline with time intervals
        prev_timestamp = None
        for event in events:
            # Insert time interval token if needed
            if include_time_intervals and prev_timestamp is not None:
                gap = event.timestamp - prev_timestamp
                interval_token = self.time_interval_tokenizer.tokenize_gap(gap)
                if interval_token:
                    result['timeline_tokens'].append(interval_token)

            # Add event tokens
            result['timeline_tokens'].append(event.tokens)
            prev_timestamp = event.timestamp

        # Build full sequence string
        result['full_sequence'] = ' '.join(result['timeline_tokens'])

        return result

    def get_combined_vocabulary(self) -> dict[str, int]:
        """Get the combined vocabulary from all tokenizers."""
        combined = {}
        idx = 0

        for vocab in [
            self.diagnosis_tokenizer.vocabulary,
            self.procedure_tokenizer.vocabulary,
            self.medication_tokenizer.vocabulary,
            self.lab_tokenizer.vocabulary,
            self.bp_tokenizer.vocabulary,
            self.demography_tokenizer.vocabulary,
            self.time_interval_tokenizer.vocabulary,
        ]:
            for token in vocab:
                if token not in combined:
                    combined[token] = idx
                    idx += 1

        return combined

    def print_session_timeline(
        self,
        hadm_id: int,
        tables: dict[str, pd.DataFrame],
        max_events: int | None = None,
    ) -> None:
        """
        Print a human-readable timeline for a session.
        """
        result = self.tokenize_session(hadm_id, tables)

        print(f"=" * 80)
        print(f"PATIENT TIMELINE - Admission {hadm_id}")
        print(f"=" * 80)
        print(f"Subject ID:     {result['subject_id']}")
        print(f"Admit Time:     {result['admit_time']}")
        print(f"Discharge Time: {result['discharge_time']}")
        print(f"Total Events:   {len(result['events'])}")
        print()

        if 'demography_tokens' in result:
            print(f"[DEMOGRAPHY] {result['demography_tokens']}")
            print()

        events = result['events']
        if max_events and len(events) > max_events:
            events = events[:max_events]
            print(f"(Showing first {max_events} events)")
            print()

        prev_timestamp = None
        for event in events:
            # Show time gap
            if prev_timestamp is not None:
                gap = event.timestamp - prev_timestamp
                interval_token = self.time_interval_tokenizer.tokenize_gap(gap)
                if interval_token:
                    print(f"    {interval_token}")

            # Show event
            print(f"[{event.timestamp}] [{event.event_type}] {event.tokens}")
            prev_timestamp = event.timestamp

        print()
        print(f"=" * 80)
        print(f"FULL TOKEN SEQUENCE ({len(result['timeline_tokens'])} tokens):")
        print(f"=" * 80)
        # Print with word wrap
        seq = result['full_sequence']
        if len(seq) > 500:
            print(seq[:500] + "...")
        else:
            print(seq)
