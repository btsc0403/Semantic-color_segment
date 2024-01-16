function hueHisUpdate(hisString){
    d3.select("#hueHistogram").remove();
    
    let hisList=hisString.split(',')
    let hisBar=[];
    let width=200;
    let height=200;
    let hueHis=d3.select("#hueHis").append("svg")
    .attr('width',width)
    .attr('height', height)
    .attr("id","hueHistogram").style("margin-bottom","30px").append("g").attr("transform","translate(105,110)");
    
    let max_count=0;
    let num;
    let pi=Math.PI;
    let t=2*pi;
    let n=500;
    let outerRadius=width/2-10;
    let innerRadius=outerRadius-10;
    let dataset=d3.range(0,t,t/n);
    let total=0;
    for (let i=0;i<hisList.length;i++)
    {
      num=Number(hisList[i]);
      hisBar.push(num);
      // num=hisBar[i]
      if (num>max_count)
      {
        max_count=num;
      }
      total=total+num
    }

    var xAxis = d3.scaleLinear()
      .domain([0, 20])
      .range([0, width])
    var yAxis = d3.scaleLinear()
      .domain([0, max_count])
      .range([height,0])
    let colormap=[]
    for (let i=0;i<hisBar.length;i++)
    {
       colormap.push("hsl("+2*3*i+",100%,50%)")
    }
    console.log(hisBar)

    hueHis.selectAll('path').data(d3.range(0,t,t/n))
    .enter().append("path")
    .attr("d", d3.arc()
               .outerRadius(outerRadius)
               .innerRadius(innerRadius)
               .startAngle(function(d) {return d;})
               .endAngle(function(d) {return d+t/n*1.1}))
    .style("fill", function(d){ return d3.hsl(d*360/t,1,.5);});
    hueHis.selectAll('.barpath')
    .data(hisBar)
    .enter()
    .append("path").attr("class","barpath")
    .attr("fill", (d,i)=>{return colormap[i]})
    .attr("d", d3.arc()     // imagine your doing a part of a donut plot
      .innerRadius(function(d){return innerRadius-70*d/max_count})
      .outerRadius(function(d) { return innerRadius; })
      .startAngle(function(d,i) { return 2*pi*2*3*i/360; })
      .endAngle(function(d,i) { return 2*pi*2*3*(i+1)/360; }))


  }