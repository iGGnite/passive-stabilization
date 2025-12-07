import numpy as np
from datetime import datetime
from math import trunc

from dynamics_helper_functions import *

# WGS 84 defined coordinate system
a = 6378137
f_inverse = 298.257223563
e_squared = 6.69437999014e-3
GM = 3.986004418e14


def calendar_to_JD(calendardate: str):
    """

    :param date:
    :return:
    """
    date = datetime.strptime(calendardate, '%d-%m-%Y')
    y = date.year
    m = date.month
    d = date.day
    f = 0

    c = trunc((m-14)/12)
    JD0 = d - 32075 + trunc(1461 * (y + 4800 + c)/4)
    JD0 += trunc(367*(m - 2 - c * 12)/12)
    JD0 -= trunc(3*(trunc((y + 4900 + c)/100))/4)
    JD = JD0 + f - 0.5
    # MJD = JD - 2400000.5
    return JD#, JD0, f

def approx_GMST(JD):
    D = JD - 2451545.0 # Days since J2000 for JD
    return np.pi / 12 * (18.697375 + 24.065709824279*D)

def ECI_to_PEF(position, jd):
    """
    Inertial frame to Pseudo Earth fixed frame
    :param position:
    :param jd:
    :return:
    """
    GMST = approx_GMST(jd)
    R = np.array([
        [np.cos(GMST),  np.sin(GMST), 0],
        [-np.sin(GMST), np.cos(GMST), 0],
        [0,             0,            1]
    ])
    return R @ position

def PEF_to_inertial(position: np.ndarray, jd: float):
    """
    Pseudo Earth fixed frame to inertial frame
    :param position:
    :param jd:
    :return:
    """
    GMST = approx_GMST(jd)
    R = np.array([
        [np.cos(GMST),  np.sin(GMST), 0],
        [-np.sin(GMST), np.cos(GMST), 0],
        [0,             0,            1]
    ])
    return R.T @ position


def N(latitude):
    return a/np.sqrt(1-e_squared*(np.sin(latitude)**2))

def LLA_to_PEF(lat_lon_alt):
    if lat_lon_alt.size > 3:
        lat = lat_lon_alt[:,0]
        lon = lat_lon_alt[:,1]
        alt = lat_lon_alt[:,2]
    else:
        lat = lat_lon_alt[0]
        lon = lat_lon_alt[1]
        alt = lat_lon_alt[2]
    N_lat = N(lat)
    x = (N_lat + alt)*np.cos(lat)*np.cos(lon)
    y = (N_lat + alt)*np.cos(lat)*np.sin(lon)
    z = ((1-e_squared)*N_lat+alt)*np.sin(lat)
    coords = np.vstack((x,y,z)).T.squeeze()
    print(coords.dtype)
    return coords


def initialise_state(state_init):
    if state_init["format"] == "Kepler":
        sma = state_init["kepler_state"][0]*1000 + a
        e = 0
        i = np.deg2rad(state_init["kepler_state"][1])
        raan = np.deg2rad(0)
        aop = np.deg2rad(0)
        ta = np.deg2rad(state_init["kepler_state"][2])

        r = sma * (1 - e ** 2) / (1 + e * np.cos(ta))

        i1 = np.cos(raan) * np.cos(aop) - np.sin(raan) * np.sin(aop) * np.cos(i)
        i2 = -np.cos(raan) * np.sin(aop) - np.sin(raan) * np.cos(aop) * np.cos(i)
        m1 = np.sin(raan) * np.cos(aop) + np.cos(raan) * np.sin(aop) * np.cos(i)
        m2 = -np.sin(raan) * np.sin(aop) + np.cos(raan) * np.cos(aop) * np.cos(i)
        n1 = np.sin(aop) * np.sin(i)
        n2 = np.cos(aop) * np.sin(i)

        rotation_matrix = np.array([[i1, i2], [m1, m2], [n1, n2]])
        xi_eta = r * np.array([np.cos(ta), np.sin(ta)])
        pos = np.dot(rotation_matrix, xi_eta)
        h = np.sqrt(GM * sma * (1 - e ** 2))
        vel = GM / h * np.dot(np.array([[-i1, i2],
                                      [-m1, m2],
                                      [-n1, n2]]),
                            np.array([np.sin(ta), e + np.cos(ta)]))
        return pos, vel
    elif state_init["format"] == "ECI" or "PEF":
        if state_init["format"] == "ECI":
            ECI_state = np.array(state_init["cartesian_state"])*1000 # Change km to m units
        else:
            jd = calendar_to_JD(state_init["date"])
            ECI_state = ECI_to_PEF(np.array(state_init["cartesian_state"]),
                                   jd)
        return ECI_state[0:3], ECI_state[3:6]
    else:
        raise ValueError("Unknown state format")

def initialise_rotational_state(state_init, pos, vel):
    """
    Initialises attitude and rotational rates of the vehicle
    :param state_init: dictionary loaded from yaml file
    :param pos: position of the vehicle in the ECI frame (used to construct Local Vertical Local Horizontal frame)
    :param vel: velocity of the vehicle in the ECI frame (used to construct Local Vertical Local Horizontal frame)
    :return:
    """
    r = pos / np.linalg.norm(pos)
    v = vel / np.linalg.norm(vel)
    R_ECI_to_LVLH = np.vstack([v, np.cross(r, v), r]).T # Shows x, y, and z vectors

    if state_init["attitude"][1] == 0 and state_init["attitude"][2] == 0:  # if pitch and yaw are both exactly zero, cross product evals to NaN
        print(f"Initial attitude is 0")
        eul_angles = np.deg2rad(np.array(state_init["attitude"]) + np.array([1e-10, 1e-10, 1e-10]))
    else:
        eul_angles = np.deg2rad(np.array(state_init["attitude"])) + np.array([1e-10, 1e-10, 1e-10])
    LVLH_to_body_quat = eul_to_quat(eul_angles)
    R_LVLH_to_body = quat_to_CTM(LVLH_to_body_quat)

    R_ECI_to_body =  R_ECI_to_LVLH @ R_LVLH_to_body
    inertial_to_body_quat = CTM_to_quat(R_ECI_to_body)
    omega_ib_b = np.deg2rad(np.array(state_init["rotation_rates"]))
    return inertial_to_body_quat, omega_ib_b


