window.onload = function() {
    var containerDiv = document.getElementById("dashboards"),
        url = "https://public.tableau.com/views/CustomerAnalysisTuThanh/CustomerAnalysis",
        options = {
            hideTabs: true,
            width: 1188, 
            height: 2027,
            onFirstInteractive: function () {
                console.log("Tableau visualization has finished loading.");
            }
        };

    var viz = new tableau.Viz(containerDiv, url, options);
};