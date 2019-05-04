---
layout: post
title: "Toronto Home Prices by Neighbourhood"
subtitle: "A 3D Visualization"
author: "Weseem Ahmed "
date: "May 4, 2019"
comments: true
categories: rblogging
tags: [Mapdeck]
my_variable:: tor_price_map.html
excerpt: Here I visualize what which Toronto neighbourhoods are the most expensive with a 3D bar graph.
output: html_document 
---

## Toronto home prices by neighbourhood, using Uber's WebGL Javascrip library

<p align="center">
  <img alt="tor_price_image"
  src="{{ site.baseurl }}/img/2019-05-04-Toronto_prices.PNG"/>
</p>

Toronto is notorious for sky high prices so I decided to see just how high they reach with a 3D visualization. I use two data sources, <a href = "http://maps.library.utoronto.ca/cgi-bin/files.pl?idnum=151"> a shapefile of Toronto's neighbourhoods </a>, 
and a list of <a href = "https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/"> home prices per each
neighbourhood </a>. 

### Code

First I read in the shapefile data as a polygon (personal preference, I think it looks cooler in the plot), followed by the home price data.
Make sure to change the `Neighbourhood ID` column name to just `ID` so we can merge it with the shapefile. Next, I modify `Home.Prices` by 
dividing by 50 just so the columns we'll generate are not too high and difficult to visualize.

```r
# Need to transform coordinate system to 4326 to be compatible with WebGL.
tor_shape <- st_read(".../Toronto shapefiles") %>% st_transform(4326) 

tor_data <- read.csv(".../Toronto_econ_data.csv") 
tor_data$Home.Prices <- tor_data$Home.Prices/50

tor_mapdata <- merge(tor_shape, tor_data)
```

Once we have load in our files and merge them by `ID` and now it's ready to be plotted.

```r
mapdeck(token=key, style = mapdeck_style("dark"), pitch = 45, location = c(-79.34, 43.71), zoom = 3) %>%
  add_polygon(data = tor_mapdata, layer_id = "polygon_layer", fill_colour = "Home.Prices", elevation = "Home.Prices",
              auto_highlight = T, tooltip = "Neighbourhood")
```

### Instructions
Zoom in or out, press "ctrl" while clicking and dragging to change angle and rotation.

{% include tor_prices_map.html %}
