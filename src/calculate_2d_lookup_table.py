import numpy as np
import cubepy as cp
import json
from time import time

a = 0
b = 2*np.pi
n = 5*10**3

def integral(theta_uw):
    """
        Integrates over specified angle to compute constant value.

        Parameters
        ----------
        theta_uw : float
            angle between specific u-vector and iterated u-vector.

        Returns
        -------
        result : float
            constant value from integrand.

        """
    def integrand(theta,theta_uw):
        return np.abs(np.cos(theta)*np.cos(theta_uw-theta))
    result, _ = cp.integrate(integrand, a, b, ([theta_uw]),itermax=25)
    return result.item()

def test_integral():
    """Tests integral function for known angle."""
    theta_uw = np.pi/2
    test_integral = lambda theta:np.abs(np.cos(theta)*np.sin(theta))
    assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
    theta_uw = -np.pi/2
    assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
    
    theta_uw = np.pi
    test_integral = lambda theta:np.abs(np.cos(theta)*np.cos(theta))
    assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
    
    theta_uw = np.pi/4
    test_integral = lambda theta:np.abs(np.sqrt(2)/2*np.cos(theta)\
                                        *(np.cos(theta)+np.sin(theta)))
    assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
    
    theta_uw = np.pi/2
    test_integral = lambda theta:np.abs(1/2*(np.cos(theta)+np.sin(theta))\
                                        *(np.cos(theta)-np.sin(theta)))
    assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])

test_integral()

def generate_list():
    """
        Generates list of angles for which we integrate over to create look up tables.

        Parameters
        ----------
        None.

        Returns
        -------
        result : list of list of floats
            a two column table of angles and integral values for contant look up table.

        """
    angles = np.linspace(a,b,n)
    integrals = [integral(angle) for angle in angles]
    result = np.column_stack((angles,integrals))
    return result

def save_dict(table,filename):
    """
        

        Parameters
        ----------
        table : TYPE
            DESCRIPTION.
        filename : string
            name of file used for importing constant values.

        Returns
        -------
        None.

        """
    np.savetxt(filename,table,delimiter=",")

save_dict(generate_list(),"2d_lookup_table" + str(n) + ".txt")






    
