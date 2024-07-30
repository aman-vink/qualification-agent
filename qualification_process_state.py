from enum import Enum


class QualificationProcessState(str, Enum):
    CATEGORY_CREATED = "category_created"
    CATEGORY_REMOVED = "category_removed"
    CATEGORY_UPDATED = "category_updated"
    CATEGORY_ADDED = "category_added"
    CATEGORY_INTERNAL_VALUES_REMOVED = "category_internal_values_removed"
    CATEGORY_INTERNAL_VALUES_ADDED = "category_internal_values_added"
    CATEGORY_INTERNAL_VALUES_UPDATED = "category_internal_values_updated"
    CATEGORY_WEIGHT_UPDATED = "category_weight_updated"
    CATEGORY_WEIGHTS_UPDATED = "category_weights_updated"
    CATEGORY_WEIGHTS_CHANGED = "category_weights_changed"
    CATEGORY_INTERNAL_VALUE_WEIGHTS_CHANGED = "category_internal_value_weights_changed"
    CATEGORY_NAME_CHANGED = "category_name_changed"
    NO_CHANGE = "no_change"
    ERROR = "error"
    CONVERSATION_INITIATED = "conversation_initiated"

    def __str__(self):
        return self.value