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

"""
Implementation of Figure 2.17
"""
def successive_twist_test():
	X = SE3(transl(0, 0, 0))
	twist = Twist3([1, 0, 0], [-1 ,0 ,1])

	for angle in np.arange(-np.pi, np.pi, 0.1):
		twisted = twist.SE3(angle) * X
		print(twisted)
		twisted.plot()
	for angle in np.arange(np.pi, -np.pi, -0.1):
		twisted = twist.SE3(angle) * X
		print(twisted)
		twisted.plot()
	twist.line().plot(color='red')
	plt.show()

def twist_homog_test():
	S = Twist3([2, 3, 2], [0 ,0 ,1])
	se3 = skewa(S.A)
	print(se3) # augmented skew symmetric matrix belonging se(3)
	T = trexp(se3)
	print(T)

### Exercises ###

def adjust_limits(ax, n = None):
	if n is None:
		val = 5
	else:
		val = n
	ax.axes.set_xlim3d(left=-val, right=val) 
	ax.axes.set_ylim3d(bottom=-val, top=val) 
	ax.axes.set_zlim3d(bottom=-val, top=val) 
	
def draw_cube(map_T_cube, ax):
	ax.clear()
	adjust_limits(ax)
	yield plot_cuboid(sides=[1, 1, 1], pose=map_T_cube, filled=False, ax=ax, color='blue')


def update_pose(angles=None, S=None):
	T = SE3(transl(0,0,0))
	if S:
		total_alpha = 0.0
		alpha = 1.0
		while True:
			T = T * S.SE3(alpha)
			yield T
			total_alpha += alpha
			if abs(total_alpha) > 2*np.pi:
				alpha = alpha * -1
		
	else:
		for angle in angles:
			yield SE3.Rx(angle)
		for angle in angles:
			yield SE3.Ry(angle)
		for angle in angles:
			yield SE3.Rz(angle)

class CubeOpMode(Enum):
    ROTATION = 1
    TRANSFORM_TWIST = 2
def exercise_2_2(mode = CubeOpMode.TRANSFORM_TWIST):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d', autoscale_on=True)
	adjust_limits(ax)
	# Rotation around 3 axes separately.
	def rotate_separately_along_major_axes(ax):
		ani = animation.FuncAnimation(
		    fig, partial(draw_cube, ax=ax),
		    frames=update_pose(angles=np.linspace(0, 2*np.pi, 128)), blit=True, interval=2, repeat=False)
		plt.show()
	# Rotation along a twist.
	def transform_along_twist(ax):
		ani = animation.FuncAnimation(
		    fig, partial(draw_cube, ax=ax),
		    frames=update_pose(S=Twist3([1, 0, 0], [-1 ,0 ,1])), blit=True, interval=20, repeat=False)
		plt.show()
	if mode == CubeOpMode.ROTATION:
		rotate_separately_along_major_axes(ax)
	else:
		transform_along_twist(ax)
	

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
	

def exercise_2_3():
	
	base_link_T_gripper = CompactPose3D(np.array([0.5, 0.0, 0.0]), UnitQuaternion(0.188908, [-0.178688, -0.679176, -0.686371]))
	gripper_T_approach = CompactPose3D(np.array([0.0, 0.3, 0.0]), UnitQuaternion(0, [0, 0, -1]))
	base_link_T_approach = base_link_T_gripper * gripper_T_approach
	print(base_link_T_approach)
	base_link_T_approach.plot(frame='approach', color='green')
	approach_T_base_link = base_link_T_approach.inverse()
	print(approach_T_base_link)
	approach_T_base_link.plot(frame='base_link',color='red')
	I = approach_T_base_link * base_link_T_approach
	print(I)
	I.plot(frame='origin')
	plt.show()

def exercise_2_4():
	R = SO2(1.0)
	v = np.array([1, 2])
	origin = np.zeros(2)
	
	Rv = R*v
	SO2().plot(frame='O', color='blue')
	R.plot(frame='B', color='magenta')
	
	print(R.inv()*R)
	print(R*R.inv())
	print("v: {} vs. Rv: {}".format(v, Rv.T))
	plt.quiver(0, 0, v[0], v[1], label='v', color='green', scale=5)
	plt.quiver(0,0, Rv[0], Rv[1], label='Rv', color='red', scale=5)
	plt.legend()
	plt.title('Rotation of a 2D vector')
	plt.show()

def exercise_2_6():
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
	power_series_library = expm(A)
	rodrigues_rotation_formula = trexp(A)
	cnt = 3
	power_series_implemented = power_series(A, cnt)
	while np.allclose(power_series_implemented, rodrigues_rotation_formula) == False:
		cnt += 1
		power_series_implemented = power_series(A, cnt)
	print(rodrigues_rotation_formula)
	print(power_series_implemented)
	print("Converged after {} iterations".format(cnt))

def exercise_2_8():
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
	A = skew(v) * theta # an element of so(3)
	R = SO3(rodrigues_rotation_formula(v, theta))
	quat = UnitQuaternion(R)
	my_quat = compute_quat(R)
	print(quat)
	print(my_quat)
	th = np.linalg.norm(v) * theta
	unit_v = v / np.linalg.norm(v)
	print(angvec2tr(th, unit_v))

def exercise_2_9():
	base_link_T_gripper = CompactPose3D(np.array([0.5, 0.0, 0.0]), UnitQuaternion(0.188908, [-0.178688, -0.679176, -0.686371]))
	base_link_T_camera = CompactPose3D(np.array([0.0, 0.3, 0.8]), UnitQuaternion(-1, [0, 0, 0]))
	base_link_T_gripper.plot(frame='gripper', color='green')
	base_link_T_camera.plot(frame='camera',color='red')
	gripper_R_camera = base_link_T_gripper.R.inv() * base_link_T_camera.R
	theta1, v1 = gripper_R_camera.angvec() 
	print(theta1, v1)
	camera_R_gripper = base_link_T_camera.R.inv() * base_link_T_gripper.R
	theta2, v2 = camera_R_gripper.angvec() 
	print(theta2, v2)
	print("camera_R_gripper:\n{}".format(camera_R_gripper))
	print("gripper_R_camera:\n{}".format(gripper_R_camera))
	print(camera_R_gripper * gripper_R_camera)
	s1 = Twist3([0,0,0], v1*theta1)
	s2 = Twist3([0,0,0], v2*theta2)
	print("Twist(camera_R_gripper): {}".format(s1))
	print("Twist(gripper_R_camera): {}".format(s2))
	s2.line().plot(color='red', marker='>', ls=':')
	print(np.allclose(s1.SE3().R, gripper_R_camera.R))
	plt.show()

def exercise_2_10(record=False):
	base_link_T_gripper = CompactPose3D(np.array([2, 1, 2.5]), UnitQuaternion(0.188908, [-0.178688, -0.679176, -0.686371]))
	gripper_T_base_link = base_link_T_gripper.inverse()
	v = [2, 3, 4]
	Rv = base_link_T_gripper.rotate(v)
	print("v: {} vs. Rv: {} ".format(v, Rv, Rv))

	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d', autoscale_on=True)
	
	anim = MultiFrameAnimate(axes=ax, dims=[-3.0,3.0,-3.0, 3.0,-3.0, 3.0])
	anim.add_frame(base_link_T_gripper.SE3, color='b', frame='A', length=1.0)
	anim.add_frame(gripper_T_base_link.SE3, color='m', frame='B', length=1.0)
	def animate_frames(plot_data, animation):
		A = plot_data[0]
		B = plot_data[1]
		animation._draw(A, "A")
		animation._draw(B, "B")
	def update_frames(pose):
		pose = CompactPose3D(np.array([2, 1, 2.5]), UnitQuaternion(0.188908, [-0.178688, -0.679176, -0.686371]))
		inv_pose = base_link_T_gripper.inverse()
		s1 = pose.SE3.twist()
		s2 = inv_pose.SE3.twist()
		
		for s in np.linspace(0, 1, 100):
			A = s1.SE3(theta=s)
			B = s2.SE3(theta=s)
			yield A, B
	ani = animation.FuncAnimation(
		fig=fig,
		func=animate_frames,
		frames=update_frames(base_link_T_gripper),
		fargs=(anim, ),
		blit=False,
		interval=50,
		repeat=False,
	)
	

	plt.quiver(0, 0, 0, v[0], v[1], v[2], color='red', label='v')
	plt.quiver(0, 0, 0, Rv[0], Rv[1], Rv[2], color='green', label='Rv')
	plt.title('Transformation of a 3D vector')
	plt.legend()
	
	
	print(base_link_T_gripper * gripper_T_base_link)
	print(gripper_T_base_link * base_link_T_gripper)
	if record:
		writergif = animation.PillowWriter(fps=30) 
		ani.save('./exercise_2-10.gif', writer=writergif)
	else:
		plt.show()

def exercise_2_11():
	np.random.seed(0)
	O_R_A = SO3(UnitQuaternion.Rand().R)
	np.random.seed(1)
	O_R_B = SO3(UnitQuaternion.Rand().R)
	O_R_A.plot(frame='A')
	O_R_B.plot(frame='B', color='green')
	A_R_B = O_R_A.inv() * O_R_B
	print(A_R_B.R)
	B_R_A = A_R_B.inv()
	# Convert to ang-vec representation
	th1, w1 = A_R_B.angvec()
	th2, w2 = B_R_A.angvec()
	np.testing.assert_allclose(th1, th2)
	np.testing.assert_allclose(w1, -w2)

	# Convert to twist form
	S1 = Twist3(SE3(A_R_B))
	S2 = Twist3(SE3(B_R_A))
	S2.line().plot(color='red', marker='>', ls=':', label='twist direction')
	print(S1)
	print(S2)
	plt.legend()
	plt.show()

# TODO: Use latexify when converting to Jupyter notebook
# TODO: Show compact fractionals with slashed division.
def exercise_2_12():
	from sympy import symbols, simplify, pretty, pprint, Matrix, MatrixSymbol, Eq, solve, asin, Interval, S, EmptySet
	from sympy.calculus.util import function_range
	from spatialmath.base.transforms3d import rpy2r

	roll, pitch, yaw = symbols('theta_r, theta_p, theta_y')
	R = Matrix(simplify(rpy2r(roll, pitch, yaw)))
	R_generic = MatrixSymbol('R', 3, 3)
	v = Matrix([0, 0, 1])
	
	eq = Eq(R_generic.as_explicit(), R)
	pprint(eq)
	pprint("------------------")
	# Solve the matrix equation for pitch, following the hint.
	pitch_solutions = solve(eq, pitch, dict=True)
	pitch_rhs_final = None
	pitch_eq_final = None
	# Loop over the possible pitch-solutions. Consider that we only want positive values as the book suggests.
	for i in range(0, len(pitch_solutions)):
		pitch_eq_candidate = pitch_solutions[i][pitch]
		k = symbols('k')
		new_pitch_eq_candidate = pitch_eq_candidate.replace(R_generic[2, 0], k, map=False)
		print("{} -> ".format(i+1), end='')
		pprint(Eq(pitch, pitch_eq_candidate))
		pprint("In range:")
		interval = function_range(new_pitch_eq_candidate, k, Interval(-1, 1))

		print(pretty(interval))
		if interval.intersect(Interval(S.NegativeInfinity, 0)) is not EmptySet:
			print("This is a suitable solution for pitch. Finishing the search for valid pitch solutions!")
			pitch_rhs_final = pitch_eq_candidate
			pitch_eq_final = Eq(pitch, pitch_rhs_final)
			break
	pprint("------------------")
	eq_v2 = eq.subs({pitch: pitch_rhs_final})
	pprint("Rotation matrix after plugging in pitch:")
	pprint(eq_v2)
	pprint("------------------")
	print("Yaw Step-1:")
	yaw_eq_step_1 = Eq(eq_v2.lhs[1,0] / eq_v2.lhs[0,0], eq_v2.rhs[1, 0] / eq_v2.rhs[0, 0])
	pprint(yaw_eq_step_1)
	print("\n")
	pprint("Yaw Step-2:")
	yaw_eq_step_2 = simplify(yaw_eq_step_1)
	pprint(yaw_eq_step_2)
	print("\n")
	pprint("Yaw Step-3:")
	yaw_eq_final = Eq(yaw, solve(yaw_eq_step_2, yaw)[0])
	pprint(yaw_eq_final)
	print("\n")
	roll_eq_step_1 = Eq(eq_v2.lhs[2,1] / eq_v2.lhs[2,2], eq_v2.rhs[2, 1] / eq_v2.rhs[2, 2])
	pprint(roll_eq_step_1)
	print("\n")
	pprint("Roll Step-1:")
	pprint(roll_eq_step_1)
	roll_eq_step_2 = simplify(roll_eq_step_1)
	print("\n")
	pprint("Roll Step-2:")
	pprint(roll_eq_step_2)
	print("\n")
	pprint("Roll Step-3:")
	roll_eq_final = Eq(roll, solve(roll_eq_step_2, roll)[0])
	pprint(roll_eq_final)
	print("---Symbolic calculations finished---")
	in_pitch = np.pi / 4.0
	in_roll = np.pi / 8.0
	in_yaw = np.pi/9.0
	R_evaled = R.subs({roll: in_roll, pitch: in_pitch, yaw: in_yaw})
	roll_recv = float(roll_eq_final.subs({R_generic[2,1]: R_evaled[2,1], R_generic[2,2]: R_evaled[2,2]}).rhs.evalf())
	pitch_recv = float(pitch_eq_final.subs({R_generic[2,0]: R_evaled[2,0]}).rhs.evalf())
	yaw_recv = float(yaw_eq_final.subs({R_generic[1,0]: R_evaled[1,0], R_generic[0,0]: R_evaled[0,0]}).rhs.evalf())
	np.testing.assert_allclose(in_roll, roll_recv)
	np.testing.assert_allclose(in_pitch, pitch_recv)
	np.testing.assert_allclose(in_yaw, yaw_recv)
	print("---Round-trip tests have succesfully finished---")
	S = Twist3(SE3(SO3(np.array(R_evaled).astype(np.float64))))
	
	Rv = (R @ v).T
	Rv_ = np.array(Rv.subs({roll: in_roll, pitch: in_pitch, yaw: in_yaw})).astype(np.float64).T
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d', autoscale_on=True)
	adjust_limits(ax, 1)
	S.line().plot(color='blue', marker='>', ls=':', label='twist direction')
	plt.quiver(0, 0, 0, v[0], v[1], v[2], color='red', label='v')
	plt.quiver(0, 0, 0, Rv_[0], Rv_[1], Rv_[2], color='green', label='Rv')
	plt.legend()
	plt.show()

def exercise_2_18():
	from spatialmath.base.transforms3d import rpy2r
	i = 0
	for roll in np.linspace(-np.pi/2.0, np.pi/2.0, 20):
	    for pitch in np.linspace(-np.pi/2.0, np.pi/2.0, 20):
	        for yaw in np.linspace(-np.pi/2.0, np.pi/2.0, 20):
	            i += 1
	            r_recv, p_recv, y_recv = tr2rpy(rpy2r(roll, pitch, yaw))
	            if abs(roll - r_recv) > 1e-2 or abs(yaw - y_recv) > 1e-2 :
	                print("{} -> roll[in]: {} vs. roll[out]: {}, diff: {}".format(i, roll, r_recv, roll - r_recv))
	                print("{} -> yaw[in]: {} vs. yaw[out]: {}, diff: {}".format(i, yaw, y_recv, yaw - y_recv))  

def exercise_2_20():
    cam_z = np.array([0, 1, 0])
    cam_y = np.array([0, 0, -1])
    cam_x = np.cross(cam_z, cam_y)
    world_R_camera = SO3(np.array([cam_x, cam_y, cam_z]).T)
    world_R_camera.plot(frame='camera')
    print("Attitude from rotation matrix: {}".format(tr2rpy(world_R_camera.R)[0]))
    print("Attitude from unit quaternion: {}".format(UnitQuaternion(world_R_camera).eul()[0]))
    plt.show()
	


# TODO: Convert all exercise functions to unit-tests.
# 1. Convert prints to hardcoded matrix-equality checks.

if __name__ == '__main__':
	#1. successive_twist_test()
	#2. twist_homog_test()
	exercise_2_2()
	#exercise_2_3()
	#exercise_2_4() # Skip Exercise 2.5 as it is the same thing, but for 3D.
	#exercise_2_6()
	#exercise_2_8()
	#exercise_2_9()
	#exercise_2_10()
	#exercise_2_11()
	#exercise_2_12()
	#exercise_2_18()
	#exercise_2_20