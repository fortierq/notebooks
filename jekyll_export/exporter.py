from nbconvert.exporters.html import HTMLExporter

class JekyllExporter(HTMLExporter):
    exclude_anchor_links = False