from structures.cabinets import ICabinet
from structures.patient_statuses import PatientCondition
from structures.sequence import MedicalSequence
import time # essential for generating events

class Relation:
    @staticmethod
    def from_cabinet_to_condition(cabinet: ICabinet) -> PatientCondition:
        raise NotImplementedError
    
    @staticmethod
    def from_condition_to_cabinet(condition: PatientCondition) -> ICabinet:
        raise NotImplementedError