---
title: Canadian Cluster Map
runtime: shiny
output: html_document
---

<style type="text/css">
.main-container {
  max-width: 100%;
  max-height: 1000px;
  margin-left: auto;
  margin-right: auto;
  margin-bottom: auto;
}
</style>


```{r, include=FALSE}
library(sf)
library(dplyr)
library(lwgeom)
library(purrr)
library(rgdal)
library(RColorBrewer)
library(leaflet)
library(shiny)
library(shinythemes)
library(rmapshaper)
library(tmap)
library(ggplot2)
library(gridExtra)
library(extrafont)
library(plotly)

load("C:/Users/wesee/Documents/ICP/Cluster map/Mapping/app/map.Rdata")

```


```{r, include=FALSE}
varlist <- setdiff(names(Provinces[,-1]), "geometry")

mypal <- c("white", "#005f88")
```


```{r echo=FALSE}  

ui <- bootstrapPage(
  tags$style(type = "text/css", "html, body {width:100%;height:100%}", "#map, #bar img {
width:100%;
display:block;
margin-left:auto;
margin-right: auto;
margin-top: 20;
  }"),
  
  h1("Canada Cluster Map", align = "center", 
     style = "font-family: 'Arial Black', sans-serif;
     font-weight: 500; line-height: 1.1; 
     color: black;"),
  br(),
  br(),
  br(),
  br(),
  leafletOutput("map", height = "75%", width = "80%"),
  br(),
  hr(),
  br(),
  h1("Top Regions by Cluster", align = "center", style = "font-family: 'Arial', sans-serif;
     font-weight: 500; line-height: 1.1; 
     color: black;"),
  h2("Location Quotient and Employment", align = "center", style = "font-family: 'Arial', sans-serif;
     font-weight: 500; line-height: 1.1; 
     color: black;"),
  br(),
  absolutePanel(top = 80, left = 260, draggable = F,
                selectInput("var", label = "Select Cluster", choices = varlist, 
                            selected = "Aerospace Vehicles and Defense", 
                            width = "400px"),
                width = "400px"),
  radioButtons("geo", "Select Geographic Level", choices = list("Provinces",
                                                                "Economic Regions",
                                                                "Census Divisions"), inline = T),
  uiOutput('title'),
  br(),
  plotOutput("bar", height = "50%", width = "100%"),
  br(),
  br()
  


)

server <- (function(input, output) {
  output$map = renderLeaflet({
    if (packageVersion("tmap") >= 2.0) {
      
      tm <- tm_basemap(leaflet::providers$OpenStreetMap) +
        tm_view(set.zoom.limits = c(4,10)) +
        tm_shape(Provinces) +
        tm_fill(col = input$var, legend.show = FALSE, palette = mypal, breaks = c(0,1,2,3,4,5,6), 
                popup.vars = c("Location Quotient: " =input$var), popup.format = list(digits=3)) +
        
        tm_polygons(input$var, border.col = "black") +
        tm_shape(Economic_Regions, name = "Economic Regions") +
        tm_fill(col = input$var, legend.show = FALSE, palette = mypal, breaks = c(0,1,2,3,4,5,6), 
                popup.vars = c("Location Quotient: " = input$var), popup.format = list(digits=3)) +
        
        tm_polygons(input$var, palette = "Blues", border.col = "black") +
        tm_shape(Census_Divisions, name = "Census Divisions") +
        tm_fill(col = input$var, legend.show = T, title = "Location Quotient", palette = mypal, breaks = c(0,1,2,3,4,5,6), 
                labels = c("0 to 1", "1 to 2", "2 to 3", "3 to 4", "4 to 5", "5+"), 
                popup.vars = c("Location Quotient: " = input$var), popup.format = list(digits=3)) +
        
        tm_polygons(input$var, border.col = "black") 
    }
    else {
      tm <- tm_shape(World) +
        tm_polygons(input$var) + 
        tm_view(basemaps = "Stamen.TerrainBackground")
    }
  
    tmap_leaflet(tm) %>%
      setMaxBounds(-174, 30, -35, 77) %>%
      setView(lng = -93, lat = 59, zoom = 4) %>%
      leaflet::hideGroup(list("Economic Regions", "Census Divisions")) 
  })
  
  
  output$title <- renderUI({
    h2(input$var, align = "center", 
       style = "font-family: 'Arial', sans-serif;
       font-weight: 500; line-height: 1.1; 
       color: black;")
  })
  
  
  graph.output <- reactive({
    
    master.pr <- master.pr[order(-master.pr[, input$var]),]
    master.pr$NAME <- reorder(master.pr$NAME, master.pr[, input$var])
    master.pr <- master.pr[1:10,]
    
    emp.pr <- emp.pr[order(-emp.pr[, input$var]),]
    emp.pr$NAME <- reorder(emp.pr$NAME, emp.pr[, input$var])
    emp.pr <- emp.pr[1:10,]
    
    master.er <- master.er[order(-master.er[, input$var]),]
    master.er$NAME <- reorder(master.er$NAME, master.er[, input$var])
    master.er <- master.er[1:10,]
    
    emp.er <- emp.er[order(-emp.er[, input$var]),]
    emp.er$NAME <- reorder(emp.er$NAME, emp.er[, input$var])
    emp.er <- emp.er[1:10,]
    
    master.cd <- master.cd[order(-master.cd[, input$var]),]
    master.cd$NAME <- reorder(master.cd$NAME, master.cd[, input$var])
    master.cd <- master.cd[1:10,]

    emp.cd <- emp.cd[order(-emp.cd[, input$var]),]
    emp.cd$NAME <- reorder(emp.cd$NAME, emp.cd[, input$var])
    emp.cd <- emp.cd[1:10,]
    
    
    
    prlq <- ggplot(master.pr, aes_string("NAME", as.name(input$var))) +
     # ggtitle(input$var) +
      xlab("Province") + ylab("Location Quotient") +
      theme(text=element_text(size=20,  family="sans"), axis.ticks = element_blank(), 
            panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"),
            axis.text = element_text(face = "bold", colour = "black"), axis.title = element_text(face = "bold")) +
      geom_bar(stat = "identity", color = "#005f88", fill = "#005f88") +
      scale_y_continuous(expand = c(0,0)) +
      #theme_classic() +
      coord_flip()
    
    premp <- ggplot(emp.pr, aes_string("NAME", as.name(input$var))) +
      xlab("Province") + ylab("Employment") +
      theme(text=element_text(size=20,  family="sans"), axis.ticks = element_blank(), 
            panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"),
            axis.text = element_text(face = "bold", colour = "black"), axis.title = element_text(face = "bold")) +
      geom_bar(stat = "identity", color = "#005f88", fill = "#005f88") +
      scale_y_continuous(expand = c(0,0)) +
      #theme_classic() +
      coord_flip()
    
    erlq <- ggplot(master.er, aes_string("NAME", as.name(input$var))) +
      xlab("Economic Region") + ylab("Location Quotient") +
      theme(text=element_text(size=20,  family="sans"), axis.ticks = element_blank(), 
            panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"),
            axis.text = element_text(face = "bold", colour = "black"), axis.title = element_text(face = "bold")) +
      geom_bar(stat = "identity", color = "#005f88", fill = "#005f88") +
      scale_y_continuous(expand = c(0,0)) +
      # theme_classic() +
      coord_flip()
    
    eremp <- ggplot(emp.er, aes_string("NAME", as.name(input$var))) +
      xlab("Economic Region") + ylab("Employment") +
      theme(text=element_text(size=20,  family="sans"), axis.ticks = element_blank(), 
            panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"),
            axis.text = element_text(face = "bold", colour = "black"), axis.title = element_text(face = "bold")) +
      geom_bar(stat = "identity", color = "#005f88", fill = "#005f88") +
      scale_y_continuous(expand = c(0,0)) +
      # theme_classic() +
      coord_flip()
    
    cdlq <- ggplot(master.cd, aes_string("NAME", as.name(input$var))) +
      xlab("Census Division") + ylab("Location Quotient") +
      theme(text=element_text(size=20,  family="sans"), axis.ticks = element_blank(), 
            panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"),
            axis.text = element_text(face = "bold", colour = "black"), axis.title = element_text(face = "bold")) +
      geom_bar(stat = "identity", color = "#005f88", fill = "#005f88") +
      scale_y_continuous(expand = c(0,0)) +
      # theme_classic() +
      coord_flip()
    
    cdemp <- ggplot(emp.cd, aes_string("NAME", as.name(input$var))) +
      xlab("Census Division") + ylab("Employment") +
      theme(text=element_text(size=20,  family="sans"), axis.ticks = element_blank(), 
            panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"),
            axis.text = element_text(face = "bold", colour = "black"), axis.title = element_text(face = "bold")) +
      geom_bar(stat = "identity", color = "#005f88", fill = "#005f88") +
      scale_y_continuous(expand = c(0,0)) +
      #theme_classic() +
      coord_flip()
    
    # grid.arrange(prlq, premp, erlq, eremp, cdlq, cdemp,ncol = 2)
    
    
      if("Provinces" %in% input$geo) return(grid.arrange(prlq, premp, ncol=2))
      if("Economic Regions" %in% input$geo) return(grid.arrange(erlq, eremp, ncol=2))
      if("Census Divisions" %in% input$geo) return(grid.arrange(cdlq, cdemp, ncol=2))
      })
    
    output$bar <- renderPlot({
      dataplots = graph.output()
      print(dataplots)
                            })

})

shinyApp(ui, server, options = list(height=1000, width=1800))

```
