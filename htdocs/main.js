const upColor = '#ec0000';
const upBorderColor = '#8A0000';
const downColor = '#00da3c';
const downBorderColor = '#008F28';

let myChart = echarts.init(document.getElementById('main'));
let option;

let result;

function splitData(rawData)
{
	const categoryData=[];
	const values=[];
	for(let i=0;i<rawData.length;i++)
	{
		categoryData.push(rawData[i].splice(0,1)[0]);
		values.push(rawData[i]);
	}
	return {
		categoryData: categoryData,
		values: values
	}
}

function getColumnData(data,index)
{
	let result=[];
	for(let i=0;i<data.values.length;i++)
		result.push(data.values[i][index]);
	return result;
}

window.addEventListener('resize', function() {
	myChart.resize();
});

$("#time_range").on("click",(e)=>{
	if(e.currentTarget.checked)
	{
		$(".range2").removeClass("d-none");
		$(".range1").addClass("d-none");
	}
	else
	{
		$(".range1").removeClass("d-none");
		$(".range2").addClass("d-none");
	}
})

$("#time_range")[0].click();

$("#getBtn").on("click",async (e)=>{
	let symbol=$("#symbol").val();
	let interval=$("#interval").val();
	if($("#time_range")[0].checked)
	{
		result=await eel.getData({
			symbol: symbol,
			period: $("#period").val(),
			interval: interval
		})();
	}
	else
	{
		result=await eel.getData({
			symbol: symbol,
			start: $("#start").val(),
			end: $("#end").val(),
			interval: interval
		})();
	}
	result=splitData(result);
	console.log(result);
	option = {
		title: {
			text: `${symbol} Candlestick Chart`,
			left: 0
		},
		tooltip: {
			trigger: 'axis',
			axisPointer: {
				type: 'cross'
			}
		},
		legend: {
			data: [`${interval} K`,"MA Fast","MA Slow"]
		},
		grid: {
			left: '10%',
			right: '10%',
			bottom: '15%'
		},
		xAxis: {
			type: 'category',
			data: result.categoryData,
			boundaryGap: false,
			axisLine: { onZero: false },
			splitLine: { show: false },
			min: 'dataMin',
			max: 'dataMax'
		},
		yAxis: {
			scale: true,
			splitArea: {
				show: true
			}
		},
		dataZoom: [
			{
				type: 'inside',
				start: 50,
				end: 100
			},
			{
				show: true,
				type: 'slider',
				top: '90%',
				start: 50,
				end: 100
			}
		],
		series: [
			{
				name: `${interval} K`,
				type: 'candlestick',
				data: result.values,
				itemStyle: {
					color: upColor,
					color0: downColor,
					borderColor: upBorderColor,
					borderColor0: downBorderColor
				},
			},
			{
				name: 'MA Fast',
				type: 'line',
				data: getColumnData(result,6),
				smooth: true,
				lineStyle: {
					opacity: 0.5
				}
			},
			{
				name: 'MA Slow',
				type: 'line',
				data: getColumnData(result,7),
				smooth: true,
				lineStyle: {
					opacity: 0.5
				}
			},
		]
	};
	myChart.setOption(option);
})

$("#PCABtn").on("click",async (e)=>{
	let target=$("#target").val();
	let test_size=$("#test_size").val();
	filepath=await eel.pca({target,test_size})();
	$("#PCAImage").attr("src",filepath);
})

$("#RFBtn").on("click",async (e)=>{
	let target=$("#target").val();
	let test_size=$("#test_size").val();
	filepath=await eel.randomForest({target,test_size})();
	$("#RFImage").attr("src",filepath);
})

$("#HMBtn").on("click",async (e)=>{
	let target=$("#target").val();
	let test_size=$("#test_size").val();
	filepath=await eel.heapMap({target,test_size})();
	$("#HMImage").attr("src",filepath);
})

$("#trainBtn").on("click",async (e)=>{
	let target=$("#target").val();
	let test_size=$("#test_size").val();
	let model=$("#model").val();
	r=await eel.train({target,test_size,model})();
	console.log(r)
	let content=`${model}`;
	for(let i in r)
	{
		if(i=="model"||i=="img"||i=="all_pred"||i=='split_time') continue;
		content+=`<br/>${i}: ${r[i]}`
	}
	let temp=`
		<span class="list-group-item list-group-item-action">
			${content}
		</span>`;
	$("#history").append(temp);
	if(r.img)
	{
		for(let i of r.img)
			$(`#${i.id}`).attr("src",i.filepath);
	}
	let symbol=$("#symbol").val();
	let interval=$("#interval").val();
	for(let i=0;i<target;i++)
		r.all_pred.splice(0,0,null);
	option = {
		title: {
			text: `${symbol} Candlestick Chart`,
			left: 0
		},
		tooltip: {
			trigger: 'axis',
			axisPointer: {
				type: 'cross'
			}
		},
		legend: {
			data: [`${interval} K`,"MA Fast","MA Slow","Predict Close"]
		},
		grid: {
			left: '10%',
			right: '10%',
			bottom: '15%'
		},
		xAxis: {
			type: 'category',
			data: result.categoryData,
			boundaryGap: false,
			axisLine: { onZero: false },
			splitLine: { show: false },
			min: 'dataMin',
			max: 'dataMax'
		},
		yAxis: {
			scale: true,
			splitArea: {
				show: true
			}
		},
		dataZoom: [
			{
				type: 'inside',
				start: 50,
				end: 100
			},
			{
				show: true,
				type: 'slider',
				top: '90%',
				start: 50,
				end: 100
			}
		],
		series: [
			{
				name: `${interval} K`,
				type: 'candlestick',
				data: result.values,
				itemStyle: {
					color: upColor,
					color0: downColor,
					borderColor: upBorderColor,
					borderColor0: downBorderColor
				},
			},
			{
				name: 'MA Fast',
				type: 'line',
				data: getColumnData(result,6),
				smooth: true,
				lineStyle: {
					opacity: 0.5
				}
			},
			{
				name: 'MA Slow',
				type: 'line',
				data: getColumnData(result,7),
				smooth: true,
				lineStyle: {
					opacity: 0.5
				}
			},
			{
				name: 'Predict Close',
				type: 'line',
				data: r.all_pred,
				smooth: true,
				lineStyle: {
					opacity: 0.7
				},
				markLine: {
					symbol: ['none', 'none'],
					data: [
						{
							xAxis: r.split_time
						}
					]
				}
			},
		]
	};
	myChart.setOption(option);
})