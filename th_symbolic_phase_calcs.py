"""Symbolic matter wave phase calculator

Derives some phase expressions quoted in an accompanying manuscript.

Symbol names are chosen to be as similar as possible to the manuscript.

Running this code shows the symbolic phase expressions that we refer to in
our manuscript.

To run this code you will need to install the sympy packge:
  https://www.sympy.org/en/index.html

Tested with Python 3.8 and sympy version 1.11.1, but should work with later versions.
"""

import sympy as sy
import typing
from dataclasses import dataclass

@dataclass
class ArmState:
    phi_dynamic: 'typing.Any' = object()
    x: 'typing.Any' = object()
    p: 'typing.Any' = object()

@dataclass
class Segment:
    duration: 'typing.Any' = object()
    acceleration: 'typing.Any' = object()

def update_arm_state(initial_arm_state, segment):
    """Update the current dynamic phase, momentum and velocity of a path
    based on the acceleration and duration of a segment"""

    m, hbar = sy.symbols("m hbar")

    # define a dummy time variable for integrations:
    t_prime = sy.symbols("t_prime")

    # assume constant acceleration during segment (could be changed):
    v = initial_arm_state.p / m + segment.acceleration * t_prime
    v_final = v.subs(t_prime, segment.duration)
    x_final = (initial_arm_state.x +
        sy.integrate(v, (t_prime, 0, segment.duration)))

    # apply Eq. 1 from our manuscript:
    phi_dynamic_final = (initial_arm_state.phi_dynamic +
        m / (2 * hbar) * sy.integrate(v**2, (t_prime, 0, segment.duration)))

    return ArmState(phi_dynamic_final, x_final, m * v_final)


def calculate_final_arm_state(initial_arm_state, segments):
    """Compute the final dynamic phase, momentum and velocity of a path
    based on a list of accelerations and durations (segments)"""
    if len(segments) == 0:
        return initial_arm_state
    else:
        new_arm_state = update_arm_state(initial_arm_state, segments[0])
        return calculate_final_arm_state(new_arm_state, segments[1:])

def calculate_delta_phi(path_1, path_2):
    """Calculate the final dynamic phase difference, separation phase,
    and matter wave phase for two paths (1 and 2)"""
    z_0, u_0, m, hbar = sy.symbols("z_0 u_0 m hbar")

    # note inclusion of arbitrary initial velocity, u_0:
    initial_arm_state = ArmState(0, z_0, m * u_0)
    arm_1 = calculate_final_arm_state(initial_arm_state, path_1)
    arm_2 = calculate_final_arm_state(initial_arm_state, path_2)

    delta_phi_dynamic = arm_2.phi_dynamic - arm_1.phi_dynamic
    # apply Eq. 3 from our manuscript:
    delta_phi_separation = (arm_1.x - arm_2.x) * (arm_1.p + arm_2.p) / 2 / hbar

    @dataclass
    class Phases:
        delta_phi_dynamic: 'typing.Any' = object()
        delta_phi_separation: 'typing.Any' = object()
        delta_phi_matter: 'typing.Any' = object()

    return Phases(sy.simplify(delta_phi_dynamic),
                  sy.simplify(delta_phi_separation),
                  sy.simplify(delta_phi_dynamic + delta_phi_separation))

####################################################################################
# Specific cases:
####################################################################################

def th():
    """Compute symbolic phase expressions associated with
    Tommey and Hogan, https://dx.doi.org/gmsf2r
    """
    T_g, T_w, a_g, a_e = sy.symbols("T_g T_w a_g a_e")
    path_1 = [
        Segment(T_g, a_g),
        Segment(T_w, 0),
        Segment(T_g, a_e)
        ]

    path_2 = [
        Segment(T_g, a_e),
        Segment(T_w, 0),
        Segment(T_g, a_g)
        ]
    return calculate_delta_phi(path_1, path_2)


def test_th():
    """Check Eq. 5 of manuscript"""
    phases = th()
    print("Our Eq. 5:\n phi_matter = ", phases.delta_phi_matter.factor())
    print()
    T_g, T_w, a_g, a_e, m , hbar = sy.symbols("T_g T_w a_g a_e m hbar")
    # Eq. 5:
    expected_delta_phi_matter = (
        m / (2 * hbar) * (a_g**2 - a_e**2) * T_g**2 * (T_g + T_w))
    diff = (phases.delta_phi_matter - expected_delta_phi_matter).simplify()
    # check Eq. 5:
    assert diff == 0

def th_with_f():
    """Compute symbolic phase expressions associated with
    Tommey and Hogan, https://dx.doi.org/gmsf2r
    where accelerations are different in 2nd gradient pulse
    from the 1st pulse by a (1+f) factor
    """
    T_g, T_w, a_g, a_e, f = sy.symbols("T_g T_w a_g a_e f")
    path_1 = [
        Segment(T_g, a_g),
        Segment(T_w, 0),
        Segment(T_g, a_e * (1 + f))
        ]

    path_2 = [
        Segment(T_g, a_e),
        Segment(T_w, 0),
        Segment(T_g, a_g * (1 + f))
        ]
    return calculate_delta_phi(path_1, path_2)

def test_th_with_f():
    """Check that introduction of (1+f) adjustment factor for Step 5
    accelerations gives the same expression as Eq. 5, but with:
      (T_g + T_w)
    replaced by
      [T_g (1 + (2/3) f - (1/6) f^2) + T_w (1 + f)]
    """
    phases = th_with_f()
    print("Our Eq. 5, but with a_g(5) = (1 + f) a_g(3), etc...:\n phi_matter = ",
          phases.delta_phi_matter.factor())
    print()
    T_g, T_w, a_g, a_e, m , hbar, f = sy.symbols("T_g T_w a_g a_e m hbar f")
    expected_delta_phi_matter = (
        m / (2 * hbar) * (a_g**2 - a_e**2) * T_g**2
        * (T_g * (1 + sy.Rational(2, 3) * f - sy.Rational(1, 6) * f**2)
           + T_w * (1 + f)))
    diff = (phases.delta_phi_matter - expected_delta_phi_matter).simplify()
    assert diff == 0

def th_with_g():
    T_g, T_w, a_g, a_e, g = sy.symbols("T_g T_w a_g a_e g")
    path_1 = [
        Segment(T_g, a_g),
        Segment(T_w, 0),
        Segment(T_g * (1 + g), a_e)
        ]

    path_2 = [
        Segment(T_g, a_e),
        Segment(T_w, 0),
        Segment(T_g * (1 + g), a_g)
        ]
    return calculate_delta_phi(path_1, path_2)

def test_th_with_g():
    """Check that introduction of (1+g) adjustment factor for Step 5
    durations gives the same expression as Eq. 5, but with:
      (T_g + T_w)
    replaced by
      [T_g (1 + g - (1/6) g^3) + T_w (1 + g)]
    """

    phases = th_with_g()
    print("Our Eq. 5, but with T_g(5) = (1 + g) T_g(3):\n phi_matter = ",
          phases.delta_phi_matter.factor())
    print()
    T_g, T_w, a_g, a_e, m , hbar, g = sy.symbols("T_g T_w a_g a_e m hbar g")
    expected_delta_phi_matter = (
        m / (2 * hbar) * (a_g**2 - a_e**2) * T_g**2
        * (T_g * (1 + g - sy.Rational(1, 6) * g**3)
           + T_w * (1 + g)))
    diff = (phases.delta_phi_matter - expected_delta_phi_matter).simplify()
    assert diff == 0

###################################################################################

if __name__ == "__main__":

    # check expressions of our manuscript:
    test_th()
    test_th_with_f()
    test_th_with_g()
