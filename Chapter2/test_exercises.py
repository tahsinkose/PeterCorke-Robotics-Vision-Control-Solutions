import numpy as np
from spatialmath.base import *
from spatialmath.quaternion import *
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial
from spatialmath import SO2, SO3, SE3, UnitQuaternion, Twist3
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import logm, expm
from multi_frame_animate import MultiFrameAnimate
from enum import Enum

	

class CompactPose3D:
	def __init__(self, t = np.zeros(3), q = UnitQuaternion()):
		self.t = t
		self.q = q
	def __str__(self):
		return "Translation: " + str(self.t) + "\nQuaternion: " + str(self.q)
	def __mul__(self, other):
		translation = self.t + self.rotate(other.t)
		rotation = self.q * other.q
		return CompactPose3D(translation, rotation)
	def rotate(self, v):
		try:
			result = self.q * Quaternion.Pure(v) * self.q.conj()
			return result.v
		except ValueError:
			return np.zeros(3)
	def inverse(self):
		t_inv = self.q.conj() * Quaternion.Pure(self.t) * self.q
		return CompactPose3D(t = -t_inv.v , q = self.q.conj())
	def plot(self, **kwargs):
		self.SE3.plot(**kwargs)

	@property
	def R(self):
		return SO3(self.q.R)
	@property
	def SE3(self):
		return SE3.Rt(SO3(self.q.R), self.t)
	

def test_se3_inversion():
	
	base_link_T_gripper = CompactPose3D(np.array([0.5, 0.0, 0.0]), UnitQuaternion(0.188908, [-0.178688, -0.679176, -0.686371]))
	gripper_T_approach = CompactPose3D(np.array([0.0, 0.3, 0.0]), UnitQuaternion(0, [0, 0, -1]))
	base_link_T_approach = base_link_T_gripper * gripper_T_approach
	approach_T_base_link = base_link_T_approach.inverse()
	assert (approach_T_base_link * base_link_T_approach).SE3 == SE3()

def test_matrix_exponential():
	def power_series(A, n):
		result = np.identity(A.shape[0]) + A
		B = A
		for i in range(2,n):
			B = np.dot(A,B)
			result = result + B/np.math.factorial(i)
		return result
	v = [-1, 0, 1]
	theta = 0.5
	A = skew(v) * theta # an element of so(3)
	rodrigues_rotation_formula = trexp(A)
	cnt = 3
	power_series_implemented = power_series(A, cnt)
	while np.allclose(power_series_implemented, rodrigues_rotation_formula) == False:
		cnt += 1
		power_series_implemented = power_series(A, cnt)
	assert np.allclose(power_series_implemented, rodrigues_rotation_formula)
	assert cnt == 8

def test_so3_algebra():
	def rodrigues_rotation_formula(omega, theta):
		th = np.linalg.norm(omega) * theta # theta = ||w|| * theta
		unit_omega = omega / np.linalg.norm(omega)
		skw = skew(unit_omega)
		return np.eye(3) + np.sin(th)*skw + (1 - np.cos(th))*np.dot(skw, skw)

	def compute_quat(R):
		alpha, w = R.angvec()
		return UnitQuaternion(np.cos(alpha/2.0), w*np.sin(alpha/2.0))

	v = [2, 3, 4]
	theta = 0.5
	R = SO3(rodrigues_rotation_formula(v, theta))
	A = skew(v) * theta # an element of so(3)
	power_series_library = expm(A)

	assert UnitQuaternion(R) == UnitQuaternion(trexp(A))
	assert UnitQuaternion(R) == UnitQuaternion(power_series_library)
	assert UnitQuaternion(R) == UnitQuaternion(angvec2tr(theta, v))

def test_so3_inversion():
	base_link_T_gripper = CompactPose3D(np.array([0.5, 0.0, 0.0]), UnitQuaternion(0.188908, [-0.178688, -0.679176, -0.686371]))
	base_link_T_camera = CompactPose3D(np.array([0.0, 0.3, 0.8]), UnitQuaternion(-1, [0, 0, 0]))
	
	gripper_R_camera = base_link_T_gripper.R.inv() * base_link_T_camera.R
	theta1, v1 = gripper_R_camera.angvec() 
	camera_R_gripper = base_link_T_camera.R.inv() * base_link_T_gripper.R
	s1 = Twist3([0,0,0], v1*theta1)
	theta2, v2 = camera_R_gripper.angvec() 
	assert theta1 == theta2
	assert np.array_equal(v1, -v2)
	assert np.allclose(camera_R_gripper.R, np.transpose(gripper_R_camera.R))
	assert np.allclose(s1.SE3().R, gripper_R_camera.R)


def test_angvec():
	np.random.seed(0)
	O_R_A = SO3(UnitQuaternion.Rand().R)
	np.random.seed(1)
	O_R_B = SO3(UnitQuaternion.Rand().R)
	A_R_B = O_R_A.inv() * O_R_B
	B_R_A = A_R_B.inv()
	# Convert to ang-vec representation
	th1, w1 = A_R_B.angvec()
	th2, w2 = B_R_A.angvec()
	np.testing.assert_allclose(th1, th2)
	np.testing.assert_allclose(w1, -w2)

	# Convert to twist form
	S1 = Twist3(SE3(A_R_B))
	S2 = Twist3(SE3(B_R_A))
	assert np.array_equal(S1.line().A, -S2.line().A)