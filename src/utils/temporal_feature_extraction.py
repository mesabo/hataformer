#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/13/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import pandas as pd
from dateutil.easter import easter

def is_world_holiday(date):
    """
    Identifies basic global holidays.

    Args:
        date (datetime): A single date value.

    Returns:
        int: 1 if the date is a world holiday, 0 otherwise.
    """
    world_holidays = ["01-01", "02-14", "10-31", "12-25"]  # New Year's, Valentine's, Halloween, Christmas
    return 1 if date.strftime("%m-%d") in world_holidays else 0

def is_black_friday(date):
    """
    Identifies Black Friday, which is the fourth Friday of November.

    Args:
        date (datetime): A single date value.

    Returns:
        int: 1 if the date is Black Friday, 0 otherwise.
    """
    if date.month == 11 and date.weekday() == 4 and (23 <= date.day <= 29):
        return 1
    return 0

def add_temporal_and_holiday_features(data):
    """
    Adds basic and advanced temporal features, including global holidays, to the dataset.
cle
    Args:
        data (pd.DataFrame): Input dataset containing a 'date' column.
        add_temporal_features (bool): If True, add temporal and holiday features.

    Returns:
        pd.DataFrame: The modified DataFrame with additional temporal and holiday features.
    """
    # Ensure 'date' column is in datetime format
    data["date"] = pd.to_datetime(data["date"])

    # Basic temporal features
    data["hour"] = data["date"].dt.hour
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    # Intermediate temporal features
    data["week_of_year"] = data["date"].dt.isocalendar().week
    data["day_of_month"] = data["date"].dt.day
    data["week_of_month"] = (data["date"].dt.day - 1) // 7 + 1
    data["quarter"] = data["date"].dt.quarter
    data["business_day"] = data["date"].dt.weekday < 5
    data["is_weekend"] = data["date"].dt.dayofweek >= 5

    # Global holiday features
    #data["is_world_holiday"] = data["date"].apply(is_world_holiday)
    #data["is_easter"] = data["date"].apply(lambda x: 1 if x.date() == easter(x.year) else 0)
    #data["is_black_friday"] = data["date"].apply(is_black_friday)

    return data