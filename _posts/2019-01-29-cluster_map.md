---
title: "Cluster Map"
author: "Weseem"
date: "January 29, 2019"
output: html_document
runtime: shiny
---

```{r setup, include=FALSE}
library(tmap)
library(sf)
library(dplyr)
library(lwgeom)
library(purrr)
library(rgdal)
library(RColorBrewer)
library(leaflet)
library(shiny)
library(shinythemes)
library(htmltools)
library(htmlwidgets)
library(rmapshaper)
```

```{r}
provinces.sf <- st_read("Z:/ICP RESEARCH/Working Papers/WP 34_Clusters/Research/Weseem/Mapping/Shapefiles/Provinces", stringsAsFactors = FALSE, quiet = TRUE) %>% st_transform(4326)

er.sf <- st_read("Z:/ICP RESEARCH/Working Papers/WP 34_Clusters/Research/Weseem/Mapping/Shapefiles/Economic Regions", stringsAsFactors = FALSE, quiet = TRUE) %>% st_transform(4326)

cd.sf <- st_read("Z:/ICP RESEARCH/Working Papers/WP 34_Clusters/Research/Weseem/Mapping/Shapefiles/Census Divisions", stringsAsFactors = FALSE, quiet = TRUE) %>% st_transform(4326)

csd.sf <- st_read("Z:/ICP RESEARCH/Working Papers/WP 34_Clusters/Research/Weseem/Mapping/Shapefiles/Census Subdivisions", stringsAsFactors = FALSE, quiet = TRUE) %>% st_transform(4326)
```


```{r}
master.pr <- read.csv("Z:/ICP RESEARCH/Working Papers/WP 34_Clusters/Research/Weseem/Mapping/app/data.pr.csv", 
                      header = T, stringsAsFactors = F)

master.er <- read.csv("Z:/ICP RESEARCH/Working Papers/WP 34_Clusters/Research/Weseem/Mapping/app/data.er.csv", 
                      header = T, stringsAsFactors = F)

master.cd <- read.csv("Z:/ICP RESEARCH/Working Papers/WP 34_Clusters/Research/Weseem/Mapping/app/data.cd.csv", 
                      header = T, stringsAsFactors = F)


pr.geo <- ms_simplify(provinces.sf[,"PRNAME", "geometry"], keep = 0.1)
er.geo <- ms_simplify(er.sf[, "ERNAME", "geometry"], keep = 0.1)
cd.geo <- ms_simplify(cd.sf[, "CDNAME", "geometry"], keep = 0.1)


Provinces <- merge(master.pr, pr.geo, by = "PRNAME") %>% st_as_sf()
Economic_Regions <- merge(master.er, er.geo, by = "ERNAME") %>% st_as_sf()
Census_Divisions <- merge(master.cd, cd.geo, by = "CDNAME") %>% st_as_sf()
```

```{r}
varlist <- setdiff(names(Provinces[,-1]), "geometry")

ui <- bootstrapPage(
    #titlePanel("Cluster Map"),
    
    absolutePanel(top = 1, left = 85, draggable = T,
                  selectInput("var", label = "Select Cluster", choices = varlist, selected = "Aerospace Vehicles and Defense", width = "400px")  
                  , width = "400px"), 
    mainPanel(
      leafletOutput("map", height = 800, width = "100%")
      
    )
  )
  
server <- (function(input, output) {
    output$map = renderLeaflet({
      if (packageVersion("tmap") >= 2.0) {
        tm <- tm_basemap(leaflet::providers$OpenStreetMap) +
          tm_shape(Provinces) +
          tm_fill(col = input$var, legend.show = FALSE, palette = "Blues") +
          tm_polygons(input$var, border.col = "black") +
          tm_shape(Economic_Regions) +
          tm_fill(col = input$var, legend.show = FALSE, palette = "Blues") +
          tm_polygons(input$var, palette = "Blues", border.col = "black") +
          tm_shape(Census_Divisions) +
          tm_fill(col = input$var, legend.show = FALSE, palette = "Blues") +
          tm_polygons(input$var, palette = "Blues", border.col = "black") 
        
      } else {
        tm <- tm_shape(Provinces) +
          tm_polygons(input$var) + 
          tm_view(basemaps = "OSM")
      }
      
      tmap_leaflet(tm) %>%
        setMaxBounds(-174, 30, -35, 77) %>%
        setView(lng = -89, lat = 61, zoom = 4) %>%
        
        leaflet::hideGroup(list("Economic_Regions", "Census_Divisions")) 
    })
  })
shinyApp(ui, server)
```

