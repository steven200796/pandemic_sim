# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from dataclasses import dataclass

from ..interfaces import NonEssentialBusinessLocationState, ContactRate, ContactRateMatrix, SimTimeTuple, \
    AgeRestrictedBusinessBaseLocation, NonEssentialBusinessBaseLocation

__all__ = ['Bar', 'Restaurant', 'RestaurantState', 'BarState']


@dataclass
class RestaurantState(NonEssentialBusinessLocationState):
    contact_rate: ContactRate = ContactRate(1, 1, 0, 0.3, 0.35, 0.1)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(11, 16)) + tuple(range(19, 24)),
                                           week_days=tuple(range(1, 7)))


    # Chef, waiter, diner
    # Rationale:
    # Chefs work closely with other chefs, are fairly separated from the waiters, but touch the food of all diners
    # Waiters have some exposure to other waiters, and more exposure to the diners
    # Diners have minimal exposure to other diners, pending seating arrangements
    """
    contact_rate_matrix = [
            [0.8, 0.2, 1],
            [0.2, 0.6, 0.8],
            [1, 0.8, 0.3]
            ]
            """

    contact_rate_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
            ]
    # 20% chefs 80% waiter breakdown for assignees
    assignee_roles = [0.2, 0.8]
    # all visitors are diners
    visitor_roles = [1]
    contact_rate_matrix: ContactRateMatrix = ContactRateMatrix(fraction_contact_matrix = contact_rate_matrix, assignee_roles = assignee_roles, visitor_roles = visitor_roles)


class Restaurant(NonEssentialBusinessBaseLocation[RestaurantState]):
    """Implements a restaurant location."""
    state_type = RestaurantState


@dataclass
class BarState(NonEssentialBusinessLocationState):
    contact_rate: ContactRate = ContactRate(1, 1, 0, 0.7, 0.2, 0.1)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(21, 24)), week_days=tuple(range(1, 7)))


class Bar(AgeRestrictedBusinessBaseLocation[BarState]):
    """Implements a Bar"""
    state_type = BarState
    age_limits = (21, 110)
