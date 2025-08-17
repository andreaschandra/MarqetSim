clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -r marqetsim.egg-info
	rm marqetsim/marqetsim.egg-info