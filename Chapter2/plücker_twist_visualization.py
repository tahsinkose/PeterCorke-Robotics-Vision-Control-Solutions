from spatialmath.geom3d import Line3
from spatialmath import SO3, SE3, Twist3
from spatialmath.base import *
import spatialmath.base as base
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import sys


def adjust_limits(ax, n = None):
    if n is None:
        val = 5
    else:
        val = n
    ax.axes.set_xlim3d(left=-val, right=val) 
    ax.axes.set_ylim3d(bottom=-val, top=val) 
    ax.axes.set_zlim3d(bottom=-val, top=val)


def show_pitch_vs_screw(record=False):
    v = np.array([2, 2, 3])
    w = np.array([-3, 1.5, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d', autoscale_on=True)
    adjust_limits(ax, 3)
    l = Line3(v, w)
    line_moment = l.v
    line_action_axis = l.uw

    plt.quiver(l.pp[0], l.pp[1], l.pp[2], 3*line_action_axis[0], 3*line_action_axis[1], 3*line_action_axis[2], 
        color='green', label=r"$\hat{l}$")
    plt.quiver(l.pp[0], l.pp[1], l.pp[2], line_moment[0], line_moment[1], line_moment[2], 
        color='magenta', label=r"$\vec{m}$ = line moment")
    l.plot(color='red', marker='>', ls=':', label='3d line')
    Q = ax.quiver(l.pp[0], l.pp[1], l.pp[2], line_moment[0], line_moment[1], line_moment[2], color='blue', label="screw moment")
    def update_pitch(k, line_action_axis, line_moment, c, payload):
        sm = np.array([line_moment[0] + k * line_action_axis[0], 
                       line_moment[1] + k * line_action_axis[1],
                       line_moment[2] + k * line_action_axis[2]])
        sm = 2*(sm / np.linalg.norm(sm))
        Q=payload[0]
        Q.remove()
        payload[0] = ax.quiver(c[0], c[1], c[2], sm[0], sm[1], sm[2], color='blue', label="screw moment")
    
    payload = [Q]
    anim = animation.FuncAnimation(fig, update_pitch, fargs=(line_action_axis, line_moment, l.pp, payload),
                               interval=10, blit=False)
    plt.legend()
    plt.title('Pitch vs. Screw Moment')
    if record:
        writergif = animation.PillowWriter(fps=30) 
        anim.save('./increase_in_pitch.gif', writer=writergif)
    else:
        plt.show()


class Plucker(Line3):
    def __init__(self, v=None, w=None):
        super().__init__(v, w)
        if w is not None:
            if np.linalg.norm(w) < 1e-4: #edge-case -> line at infinity.
                pass
            elif abs(np.linalg.norm(w) - 1) > 1e-4:
                raise ValueError('Action line vector is not unit!')
            self.data = [np.r_[v, w]]
            self.p_perp = np.cross(self.w, self.v)
    @staticmethod
    def isvalid(plucker, check=True):
        if isinstance(plucker, np.ndarray):
            plucker = Plucker(plucker[0:3], plucker[3:6])
        return abs(np.dot(plucker.v, plucker.w)) < 1e-4 and abs(np.linalg.norm(plucker.w) - 1) < 1e-4
    def __str__(self):
        return '{{{}, {}, {}; {}, {}, {}}}'.format(self.v[0], self.v[1], self.v[2], self.w[0], self.w[1], self.w[2])

    def plot_vectors(self, ax):
        ax.quiver(self.pp[0], self.pp[1], self.pp[2], self.w[0], self.w[1], self.w[2], color='green', label=r"$\hat{l}$")
        ax.quiver(self.pp[0], self.pp[1], self.pp[2], self.v[0], self.v[1], self.v[2], color='magenta', label=r"$\vec{m}$ - line moment")
        ax.quiver(self.pp[0], self.pp[1], self.pp[2], self.p_perp[0], self.p_perp[1], self.p_perp[2],
            color='blue', label=r"$\vec{p_{\perp}}$")

class Screw:
    def __init__(self, plucker: Plucker, pitch: float):
        if pitch == np.inf:
            self.s = np.zeros(3)
            self.sm = plucker.w
        else:
            self.s = plucker.w
            self.sm = plucker.v + pitch * plucker.w
    def __str__(self):
        return '{{{}, {}, {}; {}, {}, {}}}'.format(self.sm[0], self.sm[1], self.sm[2], self.s[0], self.s[1], self.s[2])
    @property
    def pitch(self):
        return np.dot(self.s, self.sm) / np.dot(self.s, self.s)

    """
    Retrieves the Plucker line of action from Screw coordinates
    """
    def ToPlucker(self):
        return Plucker(self.sm - self.pitch*self.s, self.s)

    def __eq__(s1, s2):
        return abs( 1 - np.dot(base.unitvec(np.r_[s1.s, s1.sm]), base.unitvec(np.r_[s2.s, s2.sm]))) < 1e-4
        


class Twist:
    def __init__(self, screw: Screw, theta: float):
        s_norm = np.linalg.norm(screw.s)
        if s_norm > 1e-4:
            self.t = theta * screw.s / s_norm
            self.tm = theta * screw.sm / s_norm
        else:
            self.t = np.zeros(3)
            self.tm = screw.sm
    def __str__(self):
        return '{{{}, {}, {}; {}, {}, {}}}'.format(self.tm[0], self.tm[1], self.tm[2], self.t[0], self.t[1], self.t[2])
    @property
    def pitch(self):
        return np.dot(self.t, self.tm) / pow(self.theta,2)
    @property
    def theta(self):
        return np.linalg.norm(self.t)
    @property
    def d(self):
        return self.pitch * self.theta

    def ToPlucker(self):
        if abs(self.theta) > 1e-4:
            l = self.t / self.theta
            return Plucker((self.tm / self.theta) - self.pitch * l, l)
        else:
            return Plucker(self.tm, np.zeros(3))
    @classmethod
    def FromPlucker(cls, plucker, theta, d):
        if abs(theta) > 1e-4:
            pitch = d / theta
        else:
            pitch = np.inf
        return cls(Screw(plucker, pitch), theta)

    def ToScrew(self):
        plucker = self.ToPlucker()
        return Screw(plucker, self.pitch)
    
    def SE3(self, alpha):
        if abs(self.theta) > 1e-4:
            scale_factor = alpha / self.theta
        else:
            scale_factor = alpha
        return SE3(base.trexp(np.r_[self.tm * scale_factor, self.t * scale_factor]))


    def __eq__(t1, t2):
        return abs( 1 - np.dot(base.unitvec(np.r_[t1.t, t1.tm]), base.unitvec(np.r_[t2.t, t2.tm]))) < 1e-4

    def plot_moment(self, ax):
        principal_point = self.ToPlucker().pp
        ax.quiver(principal_point[0], principal_point[1], principal_point[2], self.tm[0], self.tm[1], self.tm[2],
            color='cyan', label="twist moment", alpha=0.5)

    def plot_rotation_plane(self, ax, fig, label):
        k = np.sqrt(self.tm[0]**2 + self.tm[1]**2)
        r = np.linspace(0, k, 2)
        theta = np.linspace(0, 2*np.pi, 360)
        r, theta = np.meshgrid(r, theta)
        action_line = self.ToPlucker()
        line_moment = action_line.v
        closest_vector = action_line.p_perp
        n = np.cross(line_moment, closest_vector)
        p0 = action_line.pp
        d = -np.dot(n, p0)
        X = p0[0] + r * np.sin(theta)
        Y = p0[1] + r * np.cos(theta)
        Z = (-n[0]*X -n[1]*Y - d ) / n[2] 
        a, b = np.shape(Z)
        ax.plot_surface(X, Y, Z, alpha=0.3, color='blue')

        Q = ax.quiver(p0[0], p0[1], p0[2], self.tm[0], self.tm[1], self.tm[2],
            color='black', label=label)
        def update_body(i, p0, payload, X, Y, Z, ax):
            ax.view_init(elev=10., azim=i)
            body = payload[0]
            Q=payload[0]
            Q.remove()
            payload[0] = ax.quiver(p0[0], p0[1], p0[2], 
                    X[i, -1] - p0[0], 
                    Y[i, -1] - p0[1], 
                    Z[i, -1] - p0[2], color='black')

        payload = [Q]
        anim = animation.FuncAnimation(fig, update_body, fargs=(p0, payload, X, Y, Z, ax), frames=360,
                                  interval=10, blit=False)
        return anim


def test_conversions():
    v = np.array([2, 2, 3])
    w = np.array([-3, 1.5, 1])
    uw = w / np.linalg.norm(w)
    pitch = 0.5
    line = Plucker(v, uw)
    screw = Screw(line, pitch)
    theta = 1.5
    d = pitch * theta
    twist = Twist(screw, theta)
    assert(abs(twist.pitch - pitch) < 1e-3)
    assert(abs(twist.theta - theta) < 1e-3)
    assert(abs(twist.d - d) < 1e-3)
    assert(line == screw.ToPlucker())
    assert(line == twist.ToPlucker())
    assert(screw == twist.ToScrew())
    assert(twist == Twist.FromPlucker(line, theta, d))


def update_animation_progress(i, n):
    size=60
    def show(j):
        x = int(size*j/n)
        print("{}[{}{}] {}/{}".format("Saving frames -> ", '█'*x, "."*(size-x), j, n), 
                end='\r', file=sys.stdout, flush=True)
    show(i)

def plot_twist(theta, d, record=False):
    # Prepare scene
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111,projection='3d', autoscale_on=True)
    adjust_limits(ax, 10)

    # Prepare Plucker object
    v = np.array([2, 2, 3])
    w = np.array([-3, 1.5, 1])
    uw = w / np.linalg.norm(w)
    l = Plucker(v, uw)
    l.plot_vectors(ax)
    twist = Twist.FromPlucker(plucker = l, theta = theta, d = d)
    twist.plot_moment(ax)
    l.plot(color='red', marker='>', ls=':', label='3d line')
    plt.legend()
    plt.title(r"$\theta$ = %.2f, $d$ = %.2f"%(theta, d))
    SO3().plot(frame='O', length=0.5)

    writergif = animation.PillowWriter(fps=30)
    filename = None 
    if theta > 0.0 and abs(d) < 1e-3:
        anim = twist.plot_rotation_plane(ax, fig, "vector-under-twist" if d > 0.0 else "pure-rotation")
        filename = "pure_rotation.gif"
    else:
        filename = "twist.gif"
        # Start from the other side of action line
        X = SE3(transl(2*l.pp[0], 2*l.pp[1], 2*l.pp[2]))
        for angle in np.arange(-2*np.pi, 2*np.pi, 0.2):
            twisted = X * twist.SE3(angle) 
            twisted.plot(length=0.3, axislabel=False, alpha=0.5)
        def animate(i):
            ax.view_init(elev=10., azim=i)
            return fig,

        anim = animation.FuncAnimation(fig, animate,
                                       frames=360, interval=20, blit=False)
        
    if record:
        anim.save(filename, writer=writergif, progress_callback=update_animation_progress)
    else:
        plt.show()

plot_twist(0.0, 3.0, True)
print("\n")

#test_conversions()
#show_pitch_vs_screw(True)