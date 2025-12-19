import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.backend_bases import MouseButton

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Install with: pip install scikit-image")
    print("3D modes will be disabled.")


class UltimateConicsApp:
    def __init__(self):
        self.foci = []  # List of [x, y, sign]
        self.mode = '2D'
        self.metric = 'Euclidean'
        
        # Metric parameters
        self.p_val = 2.0      # Minkowski p
        self.angular_n = 3    # Angular: number of petals
        self.angular_a = 0.3  # Angular: amplitude
        
        self.dragging_idx = -1
        
        # Grid Resolutions
        self.res_high, self.res_low, self.res_3d = 400, 40, 40
        self._setup_grids()

        # Create figure
        self.fig = plt.figure(figsize=(14, 8))
        
        # Create BOTH 2D and 3D axes upfront, hide one
        self.ax_2d = self.fig.add_axes([0.2, 0.15, 0.75, 0.8])
        self.ax_3d = self.fig.add_axes([0.2, 0.15, 0.75, 0.8], projection='3d')
        
        # Start with 2D visible
        self.ax_3d.set_visible(False)
        self.ax = self.ax_2d
        
        self._setup_2d_axes()
        self._setup_3d_axes()
        
        self.curve_artists = []
        self.preview_artists = []
        self.focus_artists = []
        
        # Create UI elements
        self._init_ui()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        plt.show()

    def _setup_grids(self):
        l_h = np.linspace(-10, 10, self.res_high)
        self.X_h, self.Y_h = np.meshgrid(l_h, l_h)
        l_l = np.linspace(-10, 10, self.res_low)
        self.X_l, self.Y_l = np.meshgrid(l_l, l_l)
        l_3 = np.linspace(-10, 10, self.res_3d)
        self.X3, self.Y3, self.Z3 = np.meshgrid(l_3, l_3, l_3, indexing='ij')

    def _setup_2d_axes(self):
        """Configure the 2D axes."""
        self.ax_2d.set_xlim(-10, 10)
        self.ax_2d.set_ylim(-10, 10)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.grid(True, alpha=0.2)

    def _setup_3d_axes(self):
        """Configure the 3D axes."""
        self.ax_3d.set_xlim(-10, 10)
        self.ax_3d.set_ylim(-10, 10)
        self.ax_3d.set_zlim(-10, 10)

    def _switch_axes(self):
        """Switch between 2D and 3D axes by showing/hiding."""
        self.clear_artists(self.curve_artists)
        self.clear_artists(self.preview_artists)
        self.clear_artists(self.focus_artists)
        
        if self.mode == '2D':
            self.ax_3d.set_visible(False)
            self.ax_2d.set_visible(True)
            self.ax = self.ax_2d
            self.ax_2d.clear()
            self._setup_2d_axes()
        else:
            self.ax_2d.set_visible(False)
            self.ax_3d.set_visible(True)
            self.ax = self.ax_3d
            while self.ax_3d.collections:
                self.ax_3d.collections[0].remove()
            while self.ax_3d.lines:
                self.ax_3d.lines[0].remove()
            while self.ax_3d.texts:
                self.ax_3d.texts[0].remove()
            self._setup_3d_axes()
        
        self.update_foci_ui()
        self.fig.canvas.draw_idle()

    def _get_dist(self, dx, dy, dz=0):
        if self.metric == 'Euclidean':
            return np.sqrt(dx**2 + dy**2 + dz**2)
        if self.metric == 'Manhattan':
            return np.abs(dx) + np.abs(dy) + np.abs(dz)
        if self.metric == 'Chebyshev':
            return np.maximum(np.maximum(np.abs(dx), np.abs(dy)), np.abs(dz))
        if self.metric == 'Minkowski':
            p = self.p_val
            return (np.abs(dx)**p + np.abs(dy)**p + np.abs(dz)**p)**(1/p)
        if self.metric == 'Angular':
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            n = self.angular_n
            a = self.angular_a
            return r * (1 + a * np.cos(n * np.arctan2(dy, dx)))
        return 0

    def compute_field(self, grid_type='high'):
        if not self.foci:
            return None
        f_arr = np.array(self.foci)
        
        if self.mode == '2D':
            X, Y = (self.X_l, self.Y_l) if grid_type == 'low' else (self.X_h, self.Y_h)
            fx = f_arr[:, 0, None, None]
            fy = f_arr[:, 1, None, None]
            fs = f_arr[:, 2, None, None]
            return np.sum(self._get_dist(X - fx, Y - fy) * fs, axis=0)
        else:
            fx = f_arr[:, 0, None, None, None]
            fy = f_arr[:, 1, None, None, None]
            fs = f_arr[:, 2, None, None, None]
            dz = self.Z3 if self.mode == '3D Constant' else 0
            return np.sum(self._get_dist(self.X3 - fx, self.Y3 - fy, dz) * fs, axis=0)

    def auto_constant(self, event=None):
        if len(self.foci) < 1:
            return
        f_arr = np.array(self.foci)
        if len(self.foci) == 1:
            self.s_const.set_val(5.0)
        else:
            dx = f_arr[:, 0, None] - f_arr[:, 0]
            dy = f_arr[:, 1, None] - f_arr[:, 1]
            dist_matrix = np.sqrt(dx**2 + dy**2)
            max_d = np.max(dist_matrix)
            
            if any(f[2] < 0 for f in self.foci):
                self.s_const.set_val(max_d * 0.4 if max_d > 0 else 2.0)
            else:
                self.s_const.set_val(max_d * 1.2 if max_d > 0 else 5.0)
        self.generate(quality='high')

    def clear_artists(self, artist_list):
        while artist_list:
            art = artist_list.pop()
            try:
                art.remove()
            except:
                pass

    def generate(self, event=None, quality='high'):
        target_list = self.preview_artists if quality == 'low' else self.curve_artists
        self.clear_artists(target_list)
        
        if quality == 'high':
            self.clear_artists(self.preview_artists)

        field = self.compute_field(grid_type=quality)
        if field is None:
            self.fig.canvas.draw_idle()
            return

        val = self.s_const.val
        
        if self.mode == '2D':
            X, Y = (self.X_l, self.Y_l) if quality == 'low' else (self.X_h, self.Y_h)
            
            levels = sorted(set([val, -val])) if any(f[2] < 0 for f in self.foci) else [val]
            cs = self.ax.contour(X, Y, field, levels=levels,
                                 colors=['#2ca02c', '#9467bd'],
                                 linewidths=(1 if quality == 'low' else 2.5))
            
            target_list.append(cs)
        
        elif quality == 'high' and HAS_SKIMAGE:
            try:
                if self.mode == '3D Variable':
                    target = field - self.Z3
                    iso_levels = [0]
                else:
                    target = field
                    iso_levels = [val, -val] if any(f[2] < 0 for f in self.foci) else [val]
                
                cmaps = ['viridis', 'plasma']
                spacing = (20 / self.res_3d,) * 3
                
                for i, iso in enumerate(iso_levels):
                    try:
                        verts, faces, _, _ = measure.marching_cubes(target, iso, spacing=spacing)
                        verts = verts - 10
                        surf = self.ax.plot_trisurf(
                            verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces, cmap=cmaps[i % 2], alpha=0.6, lw=0
                        )
                        target_list.append(surf)
                    except (ValueError, RuntimeError):
                        pass
            except Exception as e:
                print(f"3D generation error: {e}")
        
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or self.mode != '2D':
            return
        if event.xdata is None or event.ydata is None:
            return
            
        for i, (fx, fy, _) in enumerate(self.foci):
            if np.hypot(event.xdata - fx, event.ydata - fy) < 0.6:
                if event.button == MouseButton.RIGHT:
                    self.foci[i][2] *= -1
                    self.update_foci_ui()
                    self.generate(quality='low')
                else:
                    self.dragging_idx = i
                return
                
        if event.button == MouseButton.LEFT:
            self.foci.append([event.xdata, event.ydata, 1])
            self.update_foci_ui()
            self.generate(quality='low')

    def on_motion(self, event):
        if self.dragging_idx >= 0 and event.inaxes == self.ax:
            if event.xdata is None or event.ydata is None:
                return
            self.foci[self.dragging_idx][:2] = [event.xdata, event.ydata]
            self.update_foci_ui()
            self.generate(quality='low')

    def on_release(self, event):
        if self.dragging_idx >= 0:
            self.dragging_idx = -1
            if self.mode == '2D':
                self.generate(quality='high')

    def update_foci_ui(self):
        self.clear_artists(self.focus_artists)
        
        for i, (x, y, s) in enumerate(self.foci):
            c = '#1f77b4' if s > 0 else '#d62728'
            label = f"F{i+1}{'+' if s > 0 else '-'}"
            
            if self.mode == '2D':
                scat = self.ax.scatter(x, y, c=c, s=100, edgecolors='white', zorder=10)
                txt = self.ax.text(x, y + 0.6, label, color=c, ha='center', fontweight='bold', fontsize=10)
                self.focus_artists.extend([scat, txt])
            elif self.mode == '3D Constant':
                scat = self.ax.scatter([x], [y], [0], c=c, s=120, edgecolors='white', depthshade=False)
                txt = self.ax.text(x, y, 0.8, label, color=c, fontweight='bold')
                self.focus_artists.extend([scat, txt])
            else:  # 3D Variable
                line, = self.ax.plot([x, x], [y, y], [-10, 10], c=c, ls='--', lw=2, alpha=0.7)
                txt = self.ax.text(x, y, 10.5, label, color=c, fontweight='bold')
                self.focus_artists.extend([line, txt])
        
        self.fig.canvas.draw_idle()

    def _update_metric_sliders_visibility(self):
        """Show/hide metric-specific sliders based on selected metric."""
        # Minkowski slider
        is_minkowski = (self.metric == 'Minkowski')
        self.ax_p.set_visible(is_minkowski)
        self.s_p.set_active(is_minkowski)
        
        # Angular sliders
        is_angular = (self.metric == 'Angular')
        self.ax_angular_n.set_visible(is_angular)
        self.ax_angular_a.set_visible(is_angular)
        self.s_angular_n.set_active(is_angular)
        self.s_angular_a.set_active(is_angular)
        
        self.fig.canvas.draw_idle()

    def _init_ui(self):
        """Initialize UI elements."""
        # Mode selector
        self.ax_mode = self.fig.add_axes([0.02, 0.78, 0.12, 0.12], facecolor='#f8f8f8')
        self.r_mode = RadioButtons(self.ax_mode, ('2D', '3D Constant', '3D Variable'), active=0)
        self.r_mode.on_clicked(self._change_mode)
        
        # Metric selector
        self.ax_met = self.fig.add_axes([0.02, 0.55, 0.12, 0.18], facecolor='#f8f8f8')
        self.r_met = RadioButtons(self.ax_met, ('Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski', 'Angular'), active=0)
        self.r_met.on_clicked(self._change_metric)
        
        # Buttons
        ax_auto = self.fig.add_axes([0.02, 0.46, 0.12, 0.05])
        self.b_auto = Button(ax_auto, 'Auto Constant', color='#e1f5fe')
        self.b_auto.on_clicked(self.auto_constant)
        
        ax_gen = self.fig.add_axes([0.2, 0.05, 0.14, 0.05])
        self.b_gen = Button(ax_gen, 'Generate High-Res', color='#f1f8e9')
        self.b_gen.on_clicked(lambda e: self.generate(quality='high'))
        
        ax_clr = self.fig.add_axes([0.35, 0.05, 0.08, 0.05])
        self.b_clr = Button(ax_clr, 'Clear All', color='#ffebee')
        self.b_clr.on_clicked(self._clear_all)
        
        # Main constant slider
        ax_const = self.fig.add_axes([0.5, 0.07, 0.3, 0.03])
        self.s_const = Slider(ax_const, 'Constant (K)', 0.1, 25, valinit=5)
        self.s_const.on_changed(lambda v: self.generate(quality='low'))
        
        # === METRIC-SPECIFIC SLIDERS ===
        
        # Minkowski P slider 
        self.ax_p = self.fig.add_axes([0.02, 0.40, 0.12, 0.03])
        self.s_p = Slider(self.ax_p, 'P', 1.0, 10.0, valinit=self.p_val, color='#ffcc80')
        self.s_p.on_changed(self._change_p)
        self.ax_p.set_visible(False)
        
        # Angular N slider 
        self.ax_angular_n = self.fig.add_axes([0.02, 0.34, 0.12, 0.03])
        self.s_angular_n = Slider(self.ax_angular_n, 'Petals', 1, 12, valinit=self.angular_n, valstep=1, color='#ce93d8')
        self.s_angular_n.on_changed(self._change_angular_n)
        self.ax_angular_n.set_visible(False)
        
        # Angular A slider 
        self.ax_angular_a = self.fig.add_axes([0.02, 0.28, 0.12, 0.03])
        self.s_angular_a = Slider(self.ax_angular_a, 'Amplitude', 0.05, 0.95, valinit=self.angular_a, color='#ce93d8')
        self.s_angular_a.on_changed(self._change_angular_a)
        self.ax_angular_a.set_visible(False)

    def _change_mode(self, label):
        self.mode = label
        self._switch_axes()

    def _change_metric(self, label):
        self.metric = label
        self._update_metric_sliders_visibility()
        self.ax.set_title(f"Metric: {label}")
        self.generate(quality='high')

    def _change_p(self, val):
        self.p_val = val
        if self.metric == 'Minkowski':
            self.generate(quality='low')

    def _change_angular_n(self, val):
        self.angular_n = int(val)
        if self.metric == 'Angular':
            self.generate(quality='low')

    def _change_angular_a(self, val):
        self.angular_a = val
        if self.metric == 'Angular':
            self.generate(quality='low')

    def _clear_all(self, event):
        self.foci = []
        self.clear_artists(self.curve_artists)
        self.clear_artists(self.preview_artists)
        self.clear_artists(self.focus_artists)
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    print("=" * 60)
    print("Ultimate Conics Generator")
    print("=" * 60)
    print()
    print("MODES:")
    print("  • 2D: Classic curves (circles, ellipses, hyperbolas)")
    print("  • 3D Constant: Surfaces where Σ(sign×dist) = K")
    print("  • 3D Variable: Surfaces where Σ(sign×dist) = z (cones!)")
    print()
    print("METRICS:")
    print("  • Euclidean: √(dx² + dy²) → circles, spheres")
    print("  • Manhattan: |dx| + |dy| → diamonds, octahedra")
    print("  • Chebyshev: max(|dx|, |dy|) → squares, cubes")
    print("  • Minkowski: (|dx|^p + |dy|^p)^(1/p)")
    print("      - P=1: Manhattan, P=2: Euclidean, P→∞: Chebyshev")
    print("  • Angular: r × (1 + a·cos(n·θ)) → stars, roses")
    print("      - N: number of petals (1-12)")
    print("      - A: amplitude (pointiness)")
    print()
    print("CONTROLS:")
    print("  • Left-click: Add focus (+)")
    print("  • Right-click on focus: Toggle +/−")
    print("  • Drag focus: Move it")
    print("  • Generate High-Res: Render final curve/surface")
    print("=" * 60)
    
    app = UltimateConicsApp()