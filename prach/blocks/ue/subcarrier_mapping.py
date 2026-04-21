from prach.pipeline import CommonData, Block, BlockRegistry

N_RB_SC = 12 #the number of subcarriers in one resource block;
F_S = 30_720_000 #Hz sampling rate
N_ZC_FDD = 839 # format 0-3, length of the Zadoff-Chu sequence


@BlockRegistry.register
class SubcarrierMappingBlock(Block):
	n_ul_rb : int = 6#Uplink bandwidth configuration {6; 15; 25; 50; 75; 100}
	phi: int = 7# fixed offset for formats 0-3

	n_ra_prb_offset: int = 0#RB offset to accommodate PRACH

	delta_f_ra: int = 1250 #interval between adjacent subcarriers in PRACH, format0-3
	delta_f: int = 15_000 #interval between adjacent subcarriers in LTE

	def process(self, data: CommonData) -> CommonData:
		self.n_ul_rb = data.meta.get('n_ul_rb', self.n_ul_rb) #pulling out the values
		self.n_ra_prb_offset = data.meta.get('n_ra_prb_offset', self.n_ra_prb_offset) #pulling out the values
		
		frequencies = data.meta.get("dft", []) #we get the array after dft

		K = self.delta_f/self.delta_f_ra#Zoom level - коэф масштабирования
		n_fft = int(F_S / self.delta_f_ra) # output array size (24_576)
		n_ra_prb = self.n_ra_prb_offset
		#k0 = N_RB_SC * (n_ra_prb - self.n_ul_rb / 2)
		k0 = n_ra_prb*N_RB_SC - self.n_ul_rb* N_RB_SC/2 ##
		k_start = int(n_fft // 2 + self.phi + K * (k0 +0.5))
		##first = self.phi + K * (k0 + 0.5) + 1 # from MATLAB
		##k_start = int(n_ifft // 2 + first)
		
		spectrum = [0j] * int(n_fft)

		for m in range(N_ZC_FDD):
			i = k_start + m
			if 0<=i<len(spectrum):
				spectrum[i] = frequencies[m]
			else:
				pass
		data.meta["Subcarrier_Mapping"] = spectrum

		return data 