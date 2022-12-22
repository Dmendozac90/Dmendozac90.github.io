---
layout: home
---

<div>

  <p>Hello! My name is Daniel Mendoza and I am a petroleum engineer exploring the rich and fascinating world of machine learning and data science! Check out some of the interesting projects I have worked on and feel free to contact me if you have any questions!</p>
  

  {% comment %} 
  Get categories from all posts
  {% endcomment %}
  {% assign rawtags = "" %}
  {% for post in site.posts %}
    {% assign tcategories = post.categories| join:'|' | append:'|' %}
    {% assign rawtags = rawtags | append:tcategories%}
  {% endfor %}
  {% assign rawtags = rawtags | split:'|' | sort %}

  {% comment %}
  Remove duplicate tags
  {% endcomment %}
  {% assign tags = "" %}
  {% for tag in rawtags %}
    {% if tag != "" %}
      {% if tags == "" %}
        {% assign tags = tag | split:'|' %}
      {% endif %}
      {% unless tags contains tag %}
        {% assign tags = tags | join:'|' | append:'|' | append:tag | split:'|' %}
      {% endunless %}
    {% endif %}
  {% endfor %}

  <p>
  <h2> Categories </h2>
  <a onclick="show_tag_section('all_posts')" style="cursor: pointer;" class="post_tag"> All Posts </a>
  {% for tag in tags %} 
    <a onclick="show_tag_section('{{ tag | slugify }}')" style="cursor: pointer;" class="post_tag"> {{ tag }} </a>
  {% endfor %}
  </p>

  <div id="all_posts">
  <h2> All Posts </h2>
  {% for post in site.posts %}
    <div class="post_block">
      <h3><i class="icon-chart-pie-alt"></i> - <a href="{{ post.url }}">{{ post.title }}</a></h3>
      <span><i class="icon-calendar-1"> </i><strong> - {{ post.date | date_to_string }}</strong> - {{ post.categories | array_to_sentence_string }}</span>
      {% if post.description %} 
        <p> {{ post.description }} </p>
      {% endif %}
      {% if post.img_url %} 
        <a href="{{ post.url }}" title="{{ post.title }}">
          <img src="{{ post.img_url }}" class="center_img">
        </a>
      {% endif %}
      {% if post.html_url %} 
        <a href="{{ post.url }}" title="{{ post.title }}">
          <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="{{ post.html_url }}" height="525" width="100%"></iframe>
        </a>
      {% endif %}
    </div>
  {% endfor %}
  </div>

  {% for tag in tags %}
    <div id="{{ tag | slugify }}" class="by_tag">
    <h2 id="{{ tag | slugify }}">Posts tagged "{{ tag }}"</h2>
    {% for post in site.posts %}
      {% if post.categories contains tag %}
        <div class="post_block">
          <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
          <span><strong>{{ post.date | date_to_string }}</strong> - {{ post.categories | array_to_sentence_string }}</span>
          {% if post.description %} 
            <p> {{ post.description }} </p>
          {% endif %}
          {% if post.img_url %} 
            <a href="{{ post.url }}" title="{{ post.title }}">
              <img src="{{ post.img_url }}" class="center_img">
            </a>
          {% endif %}
          {% if post.html_url %} 
                <a href="{{ post.url }}" title="{{ post.title }}">
                <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="{{ post.html_url }}" height="525" width="100%"></iframe>
                <img src="{{ post.img_url }}" width="100%">
                </a>
            {% endif %}
        </div>
      {% endif %}
    {% endfor %}
    </div>
  {% endfor %}

</div>