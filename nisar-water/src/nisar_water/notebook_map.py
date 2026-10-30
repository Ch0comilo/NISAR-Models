"""Optional notebook rectangle picker; no model or scientific runtime imports."""
from copy import deepcopy
from html import escape
import json

import ipywidgets as widgets
from ipyleaflet import DrawControl, Map, ScaleControl, TileLayer, basemaps, basemap_to_tiles

from .notebook_aoi import bbox_selection, rectangle_selection


class RectanglePicker:
    """Keep one drawn rectangle as the authoritative acquisition AOI.

    Leaflet's synchronized `data` trait covers create, edit and delete events.
    Invalid/deleted or unapplied manual input blocks selection(), preventing a
    previous/default rectangle from being used silently for a new request.
    """
    basemap_version = "esri-gibs-v1"

    def __init__(self, initial_bbox):
        initial = bbox_selection(initial_bbox)
        self.initial_bbox = initial["bbox"]
        self._selection = None
        self._updating = False
        self._manual_dirty = False
        west, south, east, north = self.initial_bbox
        self._basemap_layer = self._make_basemap("satellite")
        self.map = Map(center=((south + north) / 2, (west + east) / 2), zoom=11,
                       scroll_wheel_zoom=True, basemap=self._basemap_layer,
                       layout=widgets.Layout(height="500px", width="100%"))
        self.basemap_choice = widgets.Dropdown(description="Mapa base:", value="satellite",
            options=[("Satélite (Esri)", "satellite"), ("Topográfico (Esri)", "topographic"),
                     ("NASA Blue Marble — respaldo global", "nasa")],
            style={"description_width": "initial"}, layout=widgets.Layout(width="450px"))
        self.basemap_choice.observe(self._on_basemap, names="value")
        self.map.add(ScaleControl(position="bottomleft"))
        self.draw = DrawControl(polyline={}, polygon={}, circle={}, circlemarker={}, marker={},
                                rectangle={"shapeOptions": {"color": "#087f8c", "weight": 2,
                                                             "fillOpacity": .12}}, edit=False, remove=False)
        self.map.add(self.draw)
        self.coordinates = widgets.Text(description="O, S, E, N:", layout=widgets.Layout(width="75%"),
                                        style={"description_width": "initial"})
        self.apply = widgets.Button(description="Aplicar coordenadas", icon="check")
        self.reset = widgets.Button(description="Restaurar ejemplo", icon="undo")
        self.clear = widgets.Button(description="Borrar selección", icon="trash")
        self.status = widgets.HTML()
        instructions = widgets.HTML(
            "<b>Dibuja tu rectángulo con el botón □ del mapa.</b> Puedes mover el mapa, "
            "acercar y dibujar otro para cambiar la zona. Solo se utiliza el último. "
            "Borrar selección elimina la zona inmediatamente. "
            "El mapa base sirve de orientación; las imágenes NISAR aparecen más abajo. "
            "Si el fondo no carga, cambia Mapa base. NASA es una vista global de menor detalle.")
        self.widget = widgets.VBox([instructions, self.basemap_choice, self.map,
            widgets.HBox([self.coordinates, self.apply]), widgets.HBox([self.reset, self.clear]), self.status])
        self.draw.observe(self._on_data, names="data")
        # Leaflet's edit/delete tools retain unapplied state until Save and
        # reconstruct rectangles as editable polygons. Redraw + our immediate
        # clear/apply controls keep the displayed rectangle and kernel in sync.
        self.coordinates.observe(self._on_manual, names="value")
        self.apply.on_click(self._apply_manual)
        self.reset.on_click(lambda _: self.set_bbox(self.initial_bbox, recenter=True))
        self.clear.on_click(lambda _: self.clear_selection())
        self.set_bbox(self.initial_bbox)

    @staticmethod
    def _make_basemap(kind):
        # No requests to the community tile.openstreetmap.org service. Each
        # background uses public imagery without Earthdata credentials/keys.
        if kind == "satellite":
            layer = basemap_to_tiles(basemaps.Esri.WorldImagery, base=True)
            layer.name = "Satélite de referencia (Esri; no es NISAR)"
            return layer
        if kind == "topographic":
            layer = basemap_to_tiles(basemaps.Esri.WorldTopoMap, base=True)
            layer.name = "Topográfico (Esri)"
            return layer
        if kind == "nasa":
            provider = basemaps.NASAGIBS.BlueMarble
            return TileLayer(url=provider.build_url(), base=True, name="NASA Blue Marble (respaldo)",
                attribution=provider.html_attribution, min_zoom=0, max_zoom=20,
                max_native_zoom=provider.max_zoom)
        raise ValueError("Mapa base no admitido")

    def _on_basemap(self, change):
        layer = self._make_basemap(change["new"])
        self.map.substitute(self._basemap_layer, layer)
        self.map.basemap = layer
        self._basemap_layer = layer

    def _replace_data(self, features):
        self._updating = True
        try:
            self.draw.data = features
        finally:
            self._updating = False

    def _invalidate(self, message):
        self._selection = None
        if not self._manual_dirty:
            self._updating = True
            try:
                self.coordinates.value = ""
            finally:
                self._updating = False
        self.status.value = "<b style='color:#b42318'>Sin selección válida:</b> " + escape(message)

    def _on_data(self, change):
        if self._updating:
            return
        self._manual_dirty = False
        features = change["new"]
        if not features:
            self._invalidate("Dibuja un rectángulo antes de ejecutar la inferencia.")
            return
        feature = features[-1]
        if len(features) > 1:
            self._replace_data([feature])
        try:
            self._accept(rectangle_selection(feature))
        except ValueError as error:
            self._invalidate(str(error))

    def _accept(self, selected):
        self._selection = selected
        self._updating = True
        try:
            self.coordinates.value = json.dumps(selected["bbox"], allow_nan=False)
        finally:
            self._updating = False
        self._manual_dirty = False
        self.status.value = ("<b style='color:#087f5b'>Rectángulo seleccionado.</b> "
            "BBOX [oeste, sur, este, norte]: <code>" + escape(json.dumps(selected["bbox"])) +
            "</code><br>La búsqueda y el recorte usarán estas coordenadas, sin redondearlas. "
            "Los límites con contexto se comprueban antes de la descarga.")

    def _on_manual(self, change):
        if not self._updating:
            self._manual_dirty = True
            self.status.value = "<b>Coordenadas sin aplicar:</b> pulsa Aplicar coordenadas para cambiar el rectángulo."

    def _apply_manual(self, _):
        try:
            self.set_bbox(json.loads(self.coordinates.value), recenter=True)
        except (ValueError, TypeError) as error:
            self._invalidate(str(error))

    def set_bbox(self, bbox, *, recenter=False):
        selected = bbox_selection(bbox)
        feature = {"type": "Feature", "properties": {}, "geometry": selected["aoi"]}
        self._replace_data([feature])
        self._accept(selected)
        if recenter:
            west, south, east, north = selected["bbox"]
            self.map.center = ((south + north) / 2, (west + east) / 2)
            self.map.zoom = 11

    def clear_selection(self):
        self._replace_data([])
        self._manual_dirty = False
        self._invalidate("Dibuja un rectángulo antes de ejecutar la inferencia.")

    def selection(self):
        if self._manual_dirty:
            raise ValueError("Hay coordenadas escritas sin aplicar; pulsa Aplicar coordenadas.")
        if self._selection is None:
            raise ValueError("No hay un rectángulo válido. Dibuja o restaura uno en el mapa.")
        return deepcopy(self._selection)
