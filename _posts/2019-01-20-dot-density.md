---
layout: post
title: "Dot density map"
subtitle: "Mapping ethnicities in Toronto with ggplot2"
author: "Weseem Ahmed "
date: "January 20, 2019"
comments: true
categories: rblogging
tags: [Dot density map]
img_url: /images/20190120-dot-density.png
excerpt: This blog goes through the  steps to create a dot density map with simple features shapefiles!
output: html_document 
---

## Mapping Toronto's diversity with a dot density map
Anyone who knows me knows that visualizations are powerful tools to show data to analysts and the layman alike. And what I particularly try to achieve while visualizing data is to pack as much information as I can into a single picture. For that, maps are fantastic tools since they can show "what" "where" and "when". But what if we can show several "whats" in a single, easy to read map? That's where dot density maps come in.

### When best to use dot density maps?
-In any other normal map application situation and;
-When the variable you would like to show has several categories or groups which you would like to emphasize.


## Getting started
To begin creating our map, there are two things that we're going to need:
-The relevant shapefiles.
-Data that we want to visualize.

It's important to remember that the data we are working with must match the regions in the shapefiles. This shouldn't be a problem as long
as the sources are the same and the years are identical (ex. 2016 shapefiles with 2016 data, although close years tends to work fine but 
there will be some missing points).

For this project, I'm interested in looking where different ethnicities are located across the Toronto Census Metropolitan Area (or GTA) at
the census tract level, which means I need to download 
<a href="https://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/bound-limit-2016-eng.cfm"> census tract level shapefiles 
from Statistics Canada</a>. Make sure to select cartographic boundary file in .arc format. Next, we'll need ethnicity data by census tract
which comes from the 2016 Canadian census. For your convenience, you can download the cleaned version <a href="https://drive.google.com/file/d/1RcPDR7RN0iChuOFtT4R0gtA9tgx_z9u1/view?usp=sharing">here</a>.
With that, we may now begin our map!

<h2>Mapping dot density maps in R with ggplot2</h2>

First things first, we'll load in the libraries we'll be using for this project: 

```r
library(ggplot2)
library(sf)
library(lwgeom)
library(dplyr)
library(purrr)
library(lwgeom)
library(RColorBrewer)
```
Next, we'll load in the shapefile and convert it to simple features (which I find easier to work with than OGR). You'll notice I will be
using the pipe operator ```%>%``` often in my code. This is to save space and time by running multiple steps at once. Also keep in mind that
Statistics Canada gives us all of Canada's census tracts in one huge Canadian file, so I filter it to just get the Toronto CMA by using
```CMANAME == "Toronto"```.

```r
toronto.ct <- st_read(".../datapath/census tracts", stringsAsFactors = FALSE, quiet = TRUE) %>% 
  st_transform(4326) %>%
  filter(CMANAME == "Toronto")
```
One of the most important columns in the shapefile dataset is the ```CTUID``` variable which contains the numeric codes for every census tract
in Toronto (and really all of Canada). These have to exactly match the CTUIDs from our visible minorities dataset to be registered in properly
so we'll convert them to type numeric now:


```r
toronto.ct$CTUID <- as.numeric(toronto.ct$CTUID)
```
Now we can load in our visible minorities by using ```read.csv()```:

```r
vm.toronto <- read.csv(".../datapath/data.csv", header = TRUE, stringsAsFactors = FALSE)
attach(vm.toronto)
```
Next, perform a ```left_join()``` to merge the two datasets based on their common column ```CTUID``` and return it again as a simple features object:

```r
toronto.sf <- left_join(vm.toronto, toronto.ct) %>% st_as_sf() # Merge the shapefile and the data
attach(toronto.sf)
```
Use the following function to randomize the points within each census tract:

```r
random_round <- function(x) {
  v=as.integer(x)
  r=x-v
  test=runif(length(r), 0.0, 1.0)
  add=rep(as.integer(0),length(r))
  add[r>test] <- as.integer(1)
  value=v+add
  ifelse(is.na(value) | value<0,0,value)
  return(value)
}
```
The following command let's us control which columns (ethnicities) get plotted and what each dot represents. Since Toronto is so populous,
I opt to have each dot represent 10 people per ethnicity.

```r
num_dots <- as.data.frame(toronto.sf) %>%
select(Arab:West.Asian) %>% 
 mutate_all(funs(. / 10)) %>% 
  mutate_all(random_round)
```
Once everything is set to satisfaction, we can generate all the dots. Note this might take a couple minutes depending on how many data points
you have and the processing power of your machine:

```r
sf_dots <- map_df(names(num_dots), 
                  ~ st_sample(toronto.sf, size = num_dots[,.x], type = "random") %>% # generate the points in each polygon
                    st_cast("POINT") %>%                                             # cast the geom set as 'POINT' data
                    st_coordinates() %>%                                             # pull out coordinates into a matrix
                    as_tibble() %>%                                                  # convert to tibble
                    setNames(c("lon","lat")) %>%                                     # set column names
                    mutate(ethnicity = .x)                                           # add categorical party variable
                  ) %>% 
    slice(sample(1:n())) # once map_df binds rows randomise order to avoid bias in plotting order
```
### Plotting using ggplot2

If everything so far has gone off without a hitch, we are now ready to plot our dataset. But first, let's choose a colour palette with
```RColorBrewer```. Of course, you can choose any palette you want or even specify your own colours with hex codes.

```r
pal <- brewer.pal(n= 8, name="Set1")  
```
Last but not least, we can run ```ggplot``` and save it with ```ggsave()``` which I find works much better than the traditional way of saving.

```r
p <- ggplot() +
  geom_sf(data = toronto.sf, fill = "transparent",colour = "white") +
  geom_point(data = sf_dots, aes(lon, lat, colour = ethnicity), size = .1, alpha = 1) +
  scale_colour_manual(values = pal) +
  coord_sf(crs = 4326, datum = NA) +
  theme_void(base_family = "Iosevka", base_size = 48) +
  guides(colour = guide_legend(override.aes = list(size = 18))) +
  
  labs(x = NULL, y = NULL,
       title = "Toronto's Ethnic Mosaic",
       subtitle = "1 dot = 10 people",
       caption = "Map by @WeseemAhmed | Data Source: Statistics Canada 2016 Census") +
  
  
  theme(plot.background = element_rect(fill = "#212121", color = NA), 
        panel.background = element_rect(fill = "#212121", color = NA),
        legend.background = element_rect(fill = "#212121", color = NA),
        legend.key = element_rect(fill = "#212121", colour = NA, linetype = 'dashed'),
        legend.title = element_blank(),
        legend.position = c(0.85,0.15),
        plot.margin = margin(1, 1, 1, 1, "cm"),
        text =  element_text(color = "white"),
        title =  element_text(color = "white"),
        plot.title = element_text(hjust = 0.5),
        plot.caption = element_text(size = 32)
  )
  
  ggsave("../datapath/toronto_ethnicity.png", plot = p, dpi = 320, width = 100, height = 80, units = "cm")
```
<p align="center">
  <img alt="dot density"
  src="{{ site.baseurl }}/img/20190120-dot-density.png"/>
</p>
