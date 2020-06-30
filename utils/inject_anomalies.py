import random
import pandas as pd
from datetime import datetime, timedelta

names = ['Adam', 'Adele', 'Amanda', 'Amy', 'Anna', 'Arthur', 'Ashton', 'Barney',
        'Barry', 'Ben', 'Brian', 'Bruce', 'Caitlin', 'Cameron', 'Carla',
        'Carrie', 'Chris', 'Daniel', 'David', 'Donald', 'Denzel', 'Drew',
        'Dwight', 'Ed', 'Elizabeth', 'Ellen', 'Emil', 'Fadi', 'Frank', 'Fred',
        'George', 'Gerald', 'Gisele', 'Gordon', 'Guy', 'Hannah', 'Harper',
        'Harry', 'Henry', 'Hugo', 'Isaac', 'Ian', 'Isabella', 'Jack', 'Jake',
        'Jim', 'John', 'Junior', 'Karl', 'Kate', 'Kelly', 'Kevin', 'Lance',
        'Larry', 'Laura', 'Lewis', 'Malcolm', 'Matthew', 'Megan', 'Michelle',
        'Mick', 'Naomi', 'Natalie', 'Nicholas', 'Olivia', 'Orlando', 'Oscar',
        'Owen', 'Pamela', 'Paris', 'Paul', 'Philip', 'Quentin', 'Rachel',
        'Richard', 'Robert', 'Roger', 'Roy', 'Ryan', 'Sacha', 'Sam', 'Seth',
        'Stanley', 'Steve', 'Taylor', 'Teri', 'Thomas', 'Troy', 'Ulysses',
        'Uma', 'Vanessa', 'Vera', 'Victoria', 'Vincent', 'Wallace', 'Walter',
        'Wayne', 'Will', 'Willow', 'Zach', 'Zayn', 'Zoe']


def early_anomaly(case: pd.DataFrame) -> pd.DataFrame:
    """
    A sequence of 2 or fewer events executed too early, which is then skipped later in the case

    Parameters
    -----------------------
    case: pd.DataFrame,
        Case to apply anomaly
    Returns
    -----------------------
    Case with the applied early anomaly
    """
    case = case.reset_index(drop=True)

    timestamps = case['timestamp']
    sequence_size = random.choice([1, 2])
    if sequence_size == 1:
        original_position = random.choice(range(1, len(case)))
        activities = case.iloc[[original_position]]
        case = case.drop(original_position)
        if original_position == 1:
            anomaly_position = 0
        else:
            anomaly_position = random.choice(range(0, original_position-1))
        description = activities['activity'].values[0] + ' was originally executed at position ' + str(original_position+1) + ' and changed to position ' + str(anomaly_position+1)
    else:
        original_position = random.choice(range(1, len(case)-1))
        activities = case.iloc[original_position:original_position+2]
        case = case.drop([original_position, original_position+1])
        if original_position == 1:
            anomaly_position = 0
        else:
            anomaly_position = random.choice(range(0, original_position-1))
        description = activities['activity'].values[0] + ' and ' + activities['activity'].values[1] + ' were originally executed at positions ' + str(original_position+1) + ' and ' + str(original_position+2) + ' and changed to positions ' + str(anomaly_position+1) + ' and ' + str(anomaly_position+2)

    case = pd.concat([case.iloc[:anomaly_position], activities, case.iloc[anomaly_position:]], sort=False).reset_index(drop=True)
    case['timestamp'] = timestamps
    case['label'] = 'early'
    case['description'] = description

    return case


def late_anomaly(case: pd.DataFrame) -> pd.DataFrame:
    """
    A sequence of 2 or fewer events executed too late, which is then skipped later in the case

    Parameters
    -----------------------
    case: pd.DataFrame,
        Case to apply anomaly
    Returns
    -----------------------
    Case with the applied late anomaly
    """
    case = case.reset_index(drop=True)

    timestamps = case['timestamp']
    sequence_size = random.choice([1, 2])
    if sequence_size == 1:
        original_position = random.choice(range(0, len(case)-1))
        activities = case.iloc[[original_position]]
        case = case.drop(original_position)

        if original_position+1 == len(case):
            anomaly_position = len(case)
        else:
            anomaly_position = random.choice(range(original_position+1, len(case)))

        case = pd.concat([case.iloc[:anomaly_position], activities, case.iloc[anomaly_position:]], sort=False).reset_index(drop=True)
        description = activities['activity'].values[0] + ' was originally executed at position ' + str(original_position+1) + ' and changed to position ' + str(anomaly_position+1)
    else:
        original_position = random.choice(range(0, len(case)-2))
        activities = case.iloc[original_position:original_position+2]
        case = case.drop([original_position, original_position+1])

        if original_position+2 >= len(case):
            anomaly_position = len(case)
        else:
            anomaly_position = random.choice(range(original_position+2, len(case)))

        case = pd.concat([case.iloc[:anomaly_position-1], activities, case.iloc[anomaly_position-1:]], sort=False).reset_index(drop=True)
        description = activities['activity'].values[0] + ' and ' + activities['activity'].values[1] + ' were originally executed at positions ' + str(original_position+1) + ' and ' + str(original_position+2) + ' and changed to positions ' + str(anomaly_position) + ' and ' + str(anomaly_position+1)

    case['timestamp'] = timestamps
    case['label'] = 'late'
    case['description'] = description

    return case


def insert_anomaly(case: pd.DataFrame, random_count: int):
    """
    3 or less random activities inserted in the case

    Parameters
    -----------------------
    case: pd.DataFrame,
        Case to apply anomaly
    random_count: int,
        Integer used to complement the random activity name
    Returns
    -----------------------
    Case with the applied insert anomaly
    """
    case = case.reset_index(drop=True)
    case_id = case['case'][0]
    timestamps = list(case['timestamp'])

    k = random.choices([1, 2, 3], weights=[1, 0.5, 0.25], k=1)[0]
    anomaly_names = [f'Random Activity {i}' for i in range(random_count, random_count+k)]
    anomaly_actors = [f'Random Actor {i}' for i in range(random_count, random_count+k)]
    anomaly_resources = random.choices([10, 20, 30, 40, 50, 60, 70, 80, 90], weights=[5, 5, 5, 3, 1, 0.75, 0.5, 0.25, 0.1], k=k)
    random_count += k
    possible_positions = len(case) + k

    for name, actor, resource in zip(anomaly_names, anomaly_actors, anomaly_resources):
        activity = pd.DataFrame([[name, actor, resource]], columns=['activity', 'actor', 'resource'])
        position = random.choice(range(0, possible_positions))
        case = pd.concat([case.iloc[:position], activity, case.iloc[position:]], sort=False).reset_index(drop=True)

    timestamp_base = datetime.strptime(timestamps[-1], '%Y/%m/%d %H:%M:%S.%f')
    timestamps.extend([(timestamp_base + timedelta(minutes=i)).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3] for i in range(1, k+1)])

    case['case'] = case_id
    case['timestamp'] = timestamps
    case['label'] = 'insert'
    case['description'] = f'Activities {anomaly_names} inserted in random positions'

    return case, random_count


def skip_anomaly(case: pd.DataFrame) -> pd.DataFrame:
    """
    A sequence of 3 or less necessary events is skipped

    Parameters
    -----------------------
    case: pd.DataFrame,
        Case to apply anomaly
    Returns
    -----------------------
    Case with the applied skip anomaly
    """
    case = case.reset_index(drop=True)
    timestamps = list(case['timestamp'])

    k = random.choices([1, 2, 3], weights=[1, 0.5, 0.25], k=1)[0]
    sequence_position = random.choice(range(len(case)-k))
    positions = [i for i in range(sequence_position, sequence_position+k)]
    activities = [act for act in case.iloc[positions]['activity']]
    case = case.drop(positions)

    positions_description = [i+1 for i in positions]
    case['timestamp'] = timestamps[:len(case)]
    case['label'] = 'skip'
    if len(positions) == 1:
        case['description'] = f'Activity {activities} at position {positions_description} was skipped'
    else:
        case['description'] = f'Activities {activities} at positions {positions_description} were skipped'

    return case


def rework_anomaly(case: pd.DataFrame) -> pd.DataFrame:
    """
    A sequence of 3 or less necessary events is executed twice

    Parameters
    -----------------------
    case: pd.DataFrame,
        Case to apply anomaly
    Returns
    -----------------------
    Case with the applied rework anomaly
    """
    case = case.reset_index(drop=True)
    timestamps = list(case['timestamp'])

    k = random.choices([1, 2, 3], weights=[1, 0.5, 0.25], k=1)[0]
    sequence_position = random.choice(range(len(case)-k))
    positions = [i for i in range(sequence_position, sequence_position+k)]
    activities = [act for act in case.iloc[positions]['activity']]
    rework_activities = case.iloc[positions]
    positions_description = [i+1 for i in positions]

    case = pd.concat([case.iloc[:sequence_position], rework_activities, case.iloc[sequence_position:]], sort=False).reset_index(drop=True)

    timestamp_base = datetime.strptime(timestamps[-1], '%Y/%m/%d %H:%M:%S.%f')
    timestamps.extend([(timestamp_base + timedelta(minutes=i)).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3] for i in range(1, k+1)])
    case['timestamp'] = timestamps

    case['label'] = 'rework'
    if len(positions) == 1:
        case['description'] = f'Activity {activities} at position {positions_description} was reworked'
    else:
        case['description'] = f'Activities {activities} at positions {positions_description} were reworked'

    return case


def attribute_anomaly(case: pd.DataFrame) -> pd.DataFrame:
    """
    An incorrect attribute value is set in 3 or fewer events

    Parameters
    -----------------------
    case: pd.DataFrame,
        Case to apply anomaly
    Returns
    -----------------------
    Case with the applied attribute anomaly
    """
    case = case.reset_index(drop=True)

    k = random.choices([1, 2, 3], weights=[1, 0.5, 0.25], k=1)[0]
    positions = random.sample(range(len(case)), k=k)

    description = ''
    for position in positions:
        if random.choice([0, 1]) == 1:
            remaining = names.copy()
            original_actor = case.loc[position, 'actor']
            remaining.remove(original_actor)
            case.loc[position, 'actor'] = random.choice(remaining)
            activity = case.loc[position, 'activity']
            new_actor = case.loc[position, 'actor']
            description += f'Actor {original_actor} from {activity} was replaced by {new_actor}. '
        else:
            change_factor = k = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9], weights=[1, 3, 5, 5, 3, 1, 0.75, 0.5, 0.25], k=1)[0]
            resource_change = [i for i in range(-10*change_factor, 10*change_factor, 1)]
            resource_change.remove(0)
            resource_var = random.sample(resource_change, k=1)
            original_resource = case.loc[position, 'resource']
            case.loc[position, 'resource'] += resource_var
            activity = case.loc[position, 'activity']
            description += f'Resource {original_resource} from {activity} was complemented with {resource_var}. '

    case['label'] = 'attribute'
    case['description'] = description

    return case


def format_normal_case(case: pd.DataFrame) -> pd.DataFrame:
    """
    Returns formatted normal case

    Parameters
    -----------------------
    case: pd.DataFrame,
        Case to apply anomaly
    Returns
    -----------------------
    Case with labels and description
    """
    case = case.reset_index(drop=True)
    case['label'] = 'normal'
    case['description'] = 'normal case'
    return case
