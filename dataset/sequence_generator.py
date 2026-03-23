from structures.cabinets import ICabinet
from structures.patient_statuses import PatientCondition, is_patient_dead, is_patient_survival
from structures.sequence import MedicalSequence
from condition_cabinet_relation import Relation

class SequenceGenerator():
    def __init__(self) -> None:
        pass

    def _generate_initial_patient_condition(self) -> PatientCondition:
        raise NotImplementedError
    
    def _stop_generator(self, condition: PatientCondition) -> bool:
        return is_patient_dead(condition) or is_patient_survival(condition)

    def generate_sequence(self) -> MedicalSequence:
        sequence = MedicalSequence()

        init_patient_condition: PatientCondition = self._generate_initial_patient_condition()

        current_condition: PatientCondition = init_patient_condition

        while not self._stop_generator(current_condition):
            new_cabinet = Relation.from_condition_to_cabinet(current_condition)
            new_condition = Relation.from_cabinet_to_condition(new_cabinet)

            sequence.append_cabinet(new_cabinet)
            sequence.append_condition(new_condition)

            current_condition = new_condition

        return sequence
