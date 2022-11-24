from spatialmath.base.transforms3d import *
from spatialmath.base import animate
from collections.abc import Iterable
from collections import defaultdict
import numpy as np


class MultiFrameAnimate(animate.Animate):
	def __init__(self, axes=None, dims=None, projection="ortho", labels=("X", "Y", "Z"), **kwargs):
		super().__init__(axes, dims, projection, labels, **kwargs)
		self.displaylist = defaultdict(list)
		
	def add_frame(self, end, frame="A", start=None, **kwargs):
		self.trajectory = None
		if not isinstance(end, (np.ndarray, np.generic)) and isinstance(end, Iterable):
			try:
				if len(end) == 1:
					end = end[0]
				elif len(end) >= 2:
					self.trajectory = end
			except TypeError:
				# a generator has no len()
				self.trajectory = end

        # stash the final value
		if base.isrot(end):
			self.end = base.r2t(end)
		else:
			self.end = end

		if start is None:
			self.start = np.identity(4)
		else:
			if base.isrot(start):
				self.start = base.r2t(start)
			else:
				self.start = start
		self.current_frame = frame
        # draw axes at the origin
		base.trplot(self.start, ax=self, frame=frame, **kwargs)
	
	def _draw(self, T, frame):
		for x in self.displaylist[frame]:
			x.draw(T.A)

	"""Overload 4 proxy functions to have them work with multi-frame annotation"""
	def plot(self, x, y, z, *args, **kwargs):
		(h,) = self.ax.plot(x, y, z, *args, **kwargs)
		self.displaylist[self.current_frame].append(animate.Animate._Line(self, h, x, y, z))
		return h
	def quiver(self, x, y, z, u, v, w, *args, **kwargs):
		h = self.ax.quiver(x, y, z, u, v, w, *args, **kwargs)
		self.displaylist[self.current_frame].append(animate.Animate._Quiver(self, h))

	def text(self, x, y, z, *args, **kwargs):
		h = self.ax.text3D(x, y, z, *args, **kwargs)
		self.displaylist[self.current_frame].append(animate.Animate._Text(self, h, x, y, z))
	
	def scatter(self, xs, ys, zs, s=0, **kwargs):
		h = self.plot(xs, ys, zs, '.', markersize=0, **kwargs)
		self.displaylist[self.current_frame].append(animate.Animate._Line(self, h, xs, ys, zs))
